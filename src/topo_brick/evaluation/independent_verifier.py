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
import hmac
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    ConnectorType,
    PortPolarity,
    are_connectors_compatible,
    get_brick_primitive,
    is_known_part_id,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame, rot_id_to_matrix
from topo_brick.interface.port_manifold import AssemblyState, get_admissible_socket_pocket_ldu
from topo_brick.interface.release_provenance import (
    CatalogProvenanceError,
    PINNED_RELEASE_ASSET_DIGESTS_V5_2,
    PINNED_RELEASE_CATALOG_SHA256_BY_SCHEMA,
    PINNED_RELEASE_SET_10698_V110_SHA256,
    is_valid_sha256_hex,
    validate_compiled_manifest_dict,
)
from topo_brick.interface.se3_kinematics import (
    ContinuousSE3Pose,
    OBB3D,
    box_to_obb,
    get_connector_axial_slide_bounds_ldu,
    is_obb_protrusion_contained_in_socket_pocket,
    obb_intersects_sat,
)


CERTIFICATE_SCHEMA_VERSION: str = "1.1.1"
SUPPORTED_CERTIFICATE_SCHEMAS: Tuple[str, ...] = ("1.0.0", "1.1.0", "1.1.1")
_VALID_ASSEMBLY_MODES: Tuple[str, ...] = ("cubic_ldu", "continuous_se3")
_VALID_FOUNDATION_POLICIES: Tuple[str, ...] = (
    "clutched_baseplate",
    "table_friction_only",
    "fixed_anchor",
)


_BASE_EXTENDED_CATALOG_SHA256: Optional[str] = None
_FILE_SHA_CACHE: Dict[Path, Tuple[Tuple[int, int], str]] = {}


def _stat_cached_file_sha256(path: Path, missing_sentinel: str) -> str:
    """Compute SHA-256 over raw file bytes on disk, cached strictly by `(st_mtime_ns, st_size)` (`F7`)."""
    if not path.is_file():
        return missing_sentinel
    st = path.stat()
    sk = (int(st.st_mtime_ns), int(st.st_size))
    cached = _FILE_SHA_CACHE.get(path)
    if cached is not None and cached[0] == sk:
        return cached[1]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _FILE_SHA_CACHE[path] = (sk, digest)
    return digest


def _raw_file_sha256(path: Path, missing_sentinel: str) -> str:
    """Compute SHA-256 directly from raw disk bytes with zero stat-cache trust (`S8`)."""
    if not path.is_file():
        return missing_sentinel
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_catalog_sha256(
    schema_version: str = CERTIFICATE_SCHEMA_VERSION,
    *,
    recompute_from_disk: bool = False,
) -> str:
    """
    Compute a deterministic SHA-256 digest over raw on-disk archive bytes and in-memory primitives (`F7`, `S8`):
      - `EXTENDED_BRICK_CATALOG` (`213` curated + `TIER_A` primitives),
      - `COMPILED_LDRAW_CATALOG.archive_sha256` (`compiled_ldraw_catalog.json.gz` raw bytes),
      - `COMPILED_LDRAW_CATALOG.coacd_sha256` (`compiled_coacd_hulls.npz` raw bytes, schema `>= 1.1.0`),
      - `compiled_mold_retrieval_index.npz` raw bytes,
      - `compiled_landmark_detail_index.json.gz` raw bytes (schema `>= 1.1.0`),
      - `set_10698_1_rebrickable_inventory.json` raw bytes (schema `>= 1.1.0`).
    When `recompute_from_disk=True`, bypasses `(mtime_ns, st_size)` caches and enforces complete
    manifest provenance (`S8`).
    """
    global _BASE_EXTENDED_CATALOG_SHA256
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
        if getattr(COMPILED_LDRAW_CATALOG, "_allow_unverified_dev_catalog", False):
            raise CatalogProvenanceError(
                "Refusing certificate verification against an unverified dev catalog."
            )
        iface_dir = Path(__file__).resolve().parent.parent / "interface"
        read_fn = _raw_file_sha256 if recompute_from_disk else _stat_cached_file_sha256
        if recompute_from_disk:
            manifest_d = validate_compiled_manifest_dict(
                json.loads(COMPILED_LDRAW_CATALOG._manifest_path.read_text(encoding="utf-8")),
                require_coacd_sha256=COMPILED_LDRAW_CATALOG._coacd_npz_path.is_file(),
                require_structural_metadata=True,
            )
            cat_gz_sha = _raw_file_sha256(COMPILED_LDRAW_CATALOG._catalog_gz_path, "not_compiled")
            if not hmac.compare_digest(cat_gz_sha, str(manifest_d["catalog_gz_sha256"]).strip().lower()):
                raise CatalogProvenanceError(
                    f"Compiled catalog SHA-256 mismatch: file={cat_gz_sha} vs manifest={manifest_d['catalog_gz_sha256']}"
                )
            if COMPILED_LDRAW_CATALOG._coacd_npz_path.is_file():
                coacd_sha = _raw_file_sha256(COMPILED_LDRAW_CATALOG._coacd_npz_path, "no_coacd")
                if not hmac.compare_digest(coacd_sha, str(manifest_d["coacd_npz_sha256"]).strip().lower()):
                    raise CatalogProvenanceError(
                        f"Compiled CoACD archive SHA-256 mismatch: file={coacd_sha} vs manifest={manifest_d['coacd_npz_sha256']}"
                    )
            else:
                coacd_sha = "no_coacd"
        else:
            cat_gz_sha = COMPILED_LDRAW_CATALOG.archive_sha256
            coacd_sha = COMPILED_LDRAW_CATALOG.coacd_sha256

        ret_sha = read_fn(iface_dir / "compiled_mold_retrieval_index.npz", "no_idx")
        if schema_version == "1.0.0":
            comb_v10 = hashlib.sha256()
            comb_v10.update(
                f"{_BASE_EXTENDED_CATALOG_SHA256}:{cat_gz_sha}:{ret_sha}".encode("utf-8")
            )
            return comb_v10.hexdigest()
        lm_sha = read_fn(iface_dir / "compiled_landmark_detail_index.json.gz", "no_lm_idx")
        rb_sha = read_fn(iface_dir / "set_10698_1_rebrickable_inventory.json", "no_rb_10698")
        if schema_version == "1.1.0" and hmac.compare_digest(
            rb_sha, PINNED_RELEASE_ASSET_DIGESTS_V5_2["set_10698_1_rebrickable_inventory.json"]
        ):
            rb_sha = PINNED_RELEASE_SET_10698_V110_SHA256
        comb = hashlib.sha256()
        comb.update(
            (
                f"{_BASE_EXTENDED_CATALOG_SHA256}:{cat_gz_sha}:"
                f"{coacd_sha}:{ret_sha}:{lm_sha}:{rb_sha}"
            ).encode("utf-8")
        )
        return comb.hexdigest()
    return _BASE_EXTENDED_CATALOG_SHA256


def compute_target_sha256(target_field: Any) -> str:
    """
    Compute a canonical geometry-sensitive SHA-256 digest of `target_field` (`F7`):
    binds occupied bounds, total volume, and sorted boxes or sparse voxel tile bitmasks
    so distinct target shapes with equal bounding box and volume never collide.
    """
    if target_field is None:
        return "no_target"
    h = hashlib.sha256()
    bounds = target_field.occupied_bounds_ldu()
    vol = int(getattr(target_field, "total_volume_ldu3", 0))
    payload: Dict[str, Any] = {"bounds": bounds, "total_volume_ldu3": vol}
    if hasattr(target_field, "boxes") and getattr(target_field, "boxes", None):
        payload["boxes"] = sorted(
            (int(b.xmin), int(b.ymin), int(b.zmin), int(b.xmax), int(b.ymax), int(b.zmax))
            for b in target_field.boxes
        )
    elif hasattr(target_field, "_tiles") and isinstance(getattr(target_field, "_tiles", None), dict):
        tile_digests = []
        for tkey in sorted(target_field._tiles.keys()):
            t_arr = np.ascontiguousarray(target_field._tiles[tkey])
            tile_digests.append(
                ([int(c) for c in tkey], hashlib.sha256(t_arr.tobytes()).hexdigest()[:24])
            )
        payload["tile_digests"] = tile_digests
    elif hasattr(target_field, "occ_5x5x6"):
        occ_arr = np.ascontiguousarray(target_field.occ_5x5x6, dtype=np.uint8)
        payload["occ_5x5x6_sha256"] = hashlib.sha256(occ_arr.tobytes()).hexdigest()
    h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class VersionedCertificateArtifact:
    """Self-contained, versioned JSON certificate artifact (`R9-WS1` §3.1 + `Phase 12` `v1.1.0`)."""

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
    colored_inventory_spec: Optional[Dict[str, int]] = None
    forbidden_support_boxes: Tuple[Tuple[int, int, int, int, int, int], ...] = ()
    license_filter: str = "all"
    ground_plane_z_ldu: float = 0.0

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["placed_parts"] = list(self.placed_parts)
        d["execution_operations"] = list(self.execution_operations)
        d["matched_port_pairs"] = [
            [[int(a[0]), int(a[1])], [int(b[0]), int(b[1])]]
            for a, b in self.matched_port_pairs
        ]
        d["cached_dependencies"] = list(self.cached_dependencies)
        d["forbidden_support_boxes"] = [list(fb) for fb in self.forbidden_support_boxes]
        d["ground_plane_z_ldu"] = float(self.ground_plane_z_ldu)
        return d

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "VersionedCertificateArtifact":
        raw_gz = data.get("ground_plane_z_ldu", 0.0)
        if isinstance(raw_gz, bool) or not isinstance(raw_gz, (int, float)) or not math.isfinite(float(raw_gz)):
            raise ValueError(f"Invalid ground_plane_z_ldu: {raw_gz!r}")
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
            colored_inventory_spec=(
                {str(k): int(v) for k, v in data["colored_inventory_spec"].items()}
                if data.get("colored_inventory_spec") is not None
                else None
            ),
            forbidden_support_boxes=tuple(
                (int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]), int(fb[4]), int(fb[5]))
                for fb in data.get("forbidden_support_boxes", ())
            ),
            license_filter=str(data.get("license_filter", "all")),
            ground_plane_z_ldu=float(raw_gz),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "VersionedCertificateArtifact":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _serialize_atomic_step(
    st: Any,
    parts_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Recursively serialize an `AtomicBuildStep` into a self-contained replay operation (`F1`)."""
    sub_raw = getattr(st, "subtrace", ()) or ()
    op_str = str(st.op_type.value if hasattr(st.op_type, "value") else st.op_type)
    iids = [int(x) for x in st.instance_ids]
    part_id_s = str(st.part_id)
    rot_id_v = int(st.pose.rot_id) if st.pose is not None else 0
    rot_mat_v: Any = None
    pos_v: List[Any] = [int(c) for c in st.pose.pos] if st.pose is not None else [0, 0, 0]
    is_g_v = bool(st.is_grounded)
    col_v = int(getattr(st, "color_id", 15))
    if op_str == "PLACE_PART" and len(iids) == 1 and parts_map and iids[0] in parts_map:
        ref_p = parts_map[iids[0]]
        col_v = int(ref_p.get("color_id", col_v))
        if not part_id_s:
            part_id_s = str(ref_p["part_id"])
            rot_id_v = int(ref_p.get("rot_id", 0))
            rot_mat_v = ref_p.get("rot_matrix")
            pos_v = list(ref_p["pos"])
            is_g_v = bool(ref_p["is_grounded"])
    out_d: Dict[str, Any] = {
        "op_type": op_str,
        "instance_ids": iids,
        "part_id": part_id_s,
        "rot_id": rot_id_v,
        "pos": pos_v,
        "is_grounded": is_g_v,
        "color_id": col_v,
        "temp_support_tag": str(st.temp_support_tag),
        "gravity_unit": [float(g) for g in st.gravity_unit],
        "subtrace": [_serialize_atomic_step(sub_s, parts_map) for sub_s in sub_raw] if sub_raw else [],
    }
    if rot_mat_v is not None:
        out_d["rot_matrix"] = rot_mat_v
    return out_d


def export_certificate_artifact(
    assembly: Any,
    certificates: Any,
    validator: Any = None,
    target_field: Any = None,
    support_policy: str = "none",
    inventory_counts: Optional[Dict[str, int]] = None,
    colored_inventory: Optional[Any] = None,
    forbidden_support_boxes: Sequence[LDUBox] = (),
    license_filter: str = "all",
    foundation_policy: Optional[str] = None,
    ground_plane_z_ldu: Optional[float] = None,
) -> VersionedCertificateArtifact:
    """Export a self-contained `VersionedCertificateArtifact` (`v1.1.1`) for independent planner-free replay."""
    eff_gz = float(
        ground_plane_z_ldu
        if ground_plane_z_ldu is not None
        else getattr(certificates, "ground_plane_z_ldu", 0.0)
    )
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
                    "color_id": int(getattr(p, "color_id", 15)),
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
                    "color_id": int(getattr(p, "color_id", 15)),
                }
            )

    parts_map_tmp = {int(e["instance_id"]): e for e in placed_list}
    ops_list: List[Dict[str, Any]] = []
    eff_trace = getattr(certificates, "execution_trace", None) or getattr(assembly, "_execution_trace", None)
    if eff_trace is not None:
        for st in eff_trace.steps:
            ops_list.append(_serialize_atomic_step(st, parts_map_tmp))
    elif getattr(assembly, "has_execution_trace", False):
        step_ids = list(getattr(assembly, "recorded_step_ids", ())) or list(
            getattr(assembly, "placement_order", ())
        )
        ordered_entries = (
            [parts_map_tmp[sid] for sid in step_ids if sid in parts_map_tmp]
            if step_ids
            else placed_list
        )
        for entry in ordered_entries:
            ops_list.append(
                {
                    "op_type": "PLACE_PART",
                    "instance_ids": [int(entry["instance_id"])],
                    "part_id": str(entry["part_id"]),
                    "rot_id": int(entry.get("rot_id", 0)),
                    "rot_matrix": entry.get("rot_matrix"),
                    "pos": list(entry["pos"]),
                    "is_grounded": bool(entry["is_grounded"]),
                    "color_id": int(entry.get("color_id", 15)),
                    "temp_support_tag": "",
                    "gravity_unit": [0.0, 0.0, -1.0],
                    "subtrace": [],
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
        eq_res = validator.evaluate(
            assembly,
            foundation_policy=certificates.foundation_policy,
            ground_plane_z_ldu=eff_gz,
        )
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

    inv_cert_obj = getattr(certificates, "inventory", None)
    eff_col_src = colored_inventory if colored_inventory is not None else getattr(inv_cert_obj, "colored_inventory", None)
    eff_inv_src = inventory_counts if inventory_counts is not None else getattr(inv_cert_obj, "inventory_counts", None)

    eff_col_inv_dict: Optional[Dict[str, int]] = None
    if eff_col_src is not None:
        if hasattr(eff_col_src, "to_flat_json_dict"):
            eff_col_inv_dict = eff_col_src.to_flat_json_dict()
        elif isinstance(eff_col_src, dict):
            eff_col_inv_dict = {str(k): int(v) for k, v in eff_col_src.items()}
    eff_mold_inv = dict(eff_inv_src) if eff_inv_src is not None else None
    if eff_mold_inv is None and eff_col_src is not None and hasattr(eff_col_src, "marginal_mold_counts"):
        eff_mold_inv = eff_col_src.marginal_mold_counts()

    eff_fp_str = str(
        foundation_policy
        if foundation_policy is not None
        else (
            certificates.foundation_policy.value
            if hasattr(certificates.foundation_policy, "value")
            else certificates.foundation_policy
        )
    )
    return VersionedCertificateArtifact(
        schema_version=CERTIFICATE_SCHEMA_VERSION,
        catalog_sha256=cat_sha,
        target_sha256=tgt_sha,
        assembly_mode=mode,
        foundation_policy=eff_fp_str,
        support_policy=str(support_policy),
        inventory_spec=eff_mold_inv,
        placed_parts=tuple(placed_list),
        execution_operations=tuple(ops_list),
        matched_port_pairs=tuple(matched_pairs),
        joint_assumptions={
            "clutch_force_capacity_grams": clutch_g,
            "moment_arm_capacity_ldu": m_arm,
            "friction_coeff": fric,
            "coulomb_friction_coeff": 0.60,
            "normal_crush_ratio": 25.0,
            "locking_hinge_detent_moment_ldu_g": 380.0,
            "ball_joint_friction_moment_ldu_g": 220.0,
        },
        tolerances={
            "eq_residual_tol_g": 1e-5,
            "ub_violation_tol": 1e-5,
            "fine_csg_voxel_ldu": 2.0,
        },
        lp_witness_summary=witness_summary,
        cached_dependencies=("scipy.optimize.linprog:highs", "EXTENDED_BRICK_CATALOG"),
        claimed_certificates=certificates.summary_dict(),
        colored_inventory_spec=eff_col_inv_dict,
        forbidden_support_boxes=tuple(
            (int(fb.xmin), int(fb.ymin), int(fb.zmin), int(fb.xmax), int(fb.ymax), int(fb.zmax))
            for fb in forbidden_support_boxes
        ),
        license_filter=str(license_filter),
        ground_plane_z_ldu=eff_gz,
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
            from topo_brick.interface.se3_kinematics import ContinuousAssemblyState

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
    ground_plane_z_ldu: float = 0.0,
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
                is_stud_pair = ct_a in (
                    ConnectorType.STANDARD_STUD,
                    ConnectorType.HOLLOW_STUD,
                ) or ct_b in (
                    ConnectorType.STANDARD_STUD,
                    ConnectorType.HOLLOW_STUD,
                )
                neg_zb = (-z_b[0], -z_b[1], -z_b[2])
                if z_a != z_b and (is_stud_pair or z_a != neg_zb):
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
            if is_table and abs(z_g - float(ground_plane_z_ldu)) > 0.25:
                return False, float("inf"), float("inf"), float("inf")
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
    eq_r_idx: List[int] = []
    eq_c_idx: List[int] = []
    eq_vals: List[float] = []
    b_eq = np.zeros(n_eq, dtype=np.float64)

    coms: Dict[int, Tuple[float, float, float]] = {}
    gx_u, gy_u, gz_u = float(gravity_unit[0]), float(gravity_unit[1]), float(gravity_unit[2])
    for iid in non_ground:
        prim = get_brick_primitive(str(parts_dict[iid]["part_id"]))
        bxs = world_boxes[iid]
        tot_v = sum(b.volume_ldu3 for b in bxs)
        cx = sum(0.5 * (b.xmin + b.xmax) * b.volume_ldu3 for b in bxs) / max(tot_v, 1)
        cy = sum(0.5 * (b.ymin + b.ymax) * b.volume_ldu3 for b in bxs) / max(tot_v, 1)
        cz = sum(0.5 * (b.zmin + b.zmax) * b.volume_ldu3 for b in bxs) / max(tot_v, 1)
        coms[iid] = (float(cx), float(cy), float(cz))
        r0 = row_map[iid]
        m_g = float(prim.mass_grams)
        b_eq[r0 + 0] = -m_g * gx_u
        b_eq[r0 + 1] = -m_g * gy_u
        b_eq[r0 + 2] = -m_g * gz_u

    ub_r_idx: List[int] = []
    ub_c_idx: List[int] = []
    ub_vals: List[float] = []
    ub_rhs: List[float] = []
    bounds: List[Tuple[Optional[float], Optional[float]]] = []

    def _add_ub(cols: List[int], vals: List[float], rhs: float = 0.0) -> None:
        ri = len(ub_rhs)
        ub_rhs.append(rhs)
        for ci, vi in zip(cols, vals):
            ub_r_idx.append(ri)
            ub_c_idx.append(ci)
            ub_vals.append(vi)

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
        nx0, ny0, nz0 = float(n_v[0]), float(n_v[1]), float(n_v[2])
        u1x, u1y, u1z = float(t1_v[0]), float(t1_v[1]), float(t1_v[2])
        u2x, u2y, u2z = float(t2_v[0]), float(t2_v[1]), float(t2_v[2])
        rpx, rpy, rpz = float(r_p[0]), float(r_p[1]), float(r_p[2])
        # Action-reaction on u (-1) and v (+1): F_on_v = (f_c - f_t) n + f_u t1 + f_v t2
        for end_id, sgn in ((u, -1.0), (v, 1.0)):
            if end_id not in row_map:
                continue
            r0 = row_map[end_id]
            ccx, ccy, ccz = coms[end_id]
            drx, dry, drz = rpx - ccx, rpy - ccy, rpz - ccz
            for d_ax, nv_a, t1_a, t2_a in ((0, nx0, u1x, u2x), (1, ny0, u1y, u2y), (2, nz0, u1z, u2z)):
                if nv_a != 0.0:
                    eq_r_idx.extend((r0 + d_ax, r0 + d_ax))
                    eq_c_idx.extend((base + 0, base + 1))
                    eq_vals.extend((sgn * nv_a, -sgn * nv_a))
                if t1_a != 0.0:
                    eq_r_idx.append(r0 + d_ax)
                    eq_c_idx.append(base + 2)
                    eq_vals.append(sgn * t1_a)
                if t2_a != 0.0:
                    eq_r_idx.append(r0 + d_ax)
                    eq_c_idx.append(base + 3)
                    eq_vals.append(sgn * t2_a)
            cnx, cny, cnz = dry * nz0 - drz * ny0, drz * nx0 - drx * nz0, drx * ny0 - dry * nx0
            c1x, c1y, c1z = dry * u1z - drz * u1y, drz * u1x - drx * u1z, drx * u1y - dry * u1x
            c2x, c2y, c2z = dry * u2z - drz * u2y, drz * u2x - drx * u2z, drx * u2y - dry * u2x
            for d_ax, cn_a, c1_a, c2_a, t1_a, t2_a, nv_a in (
                (0, cnx, c1x, c2x, u1x, u2x, nx0),
                (1, cny, c1y, c2y, u1y, u2y, ny0),
                (2, cnz, c1z, c2z, u1z, u2z, nz0),
            ):
                if cn_a != 0.0:
                    eq_r_idx.extend((r0 + 3 + d_ax, r0 + 3 + d_ax))
                    eq_c_idx.extend((base + 0, base + 1))
                    eq_vals.extend((sgn * cn_a, -sgn * cn_a))
                if c1_a != 0.0:
                    eq_r_idx.append(r0 + 3 + d_ax)
                    eq_c_idx.append(base + 2)
                    eq_vals.append(sgn * c1_a)
                if c2_a != 0.0:
                    eq_r_idx.append(r0 + 3 + d_ax)
                    eq_c_idx.append(base + 3)
                    eq_vals.append(sgn * c2_a)
                if t1_a != 0.0:
                    eq_r_idx.append(r0 + 3 + d_ax)
                    eq_c_idx.append(base + 4)
                    eq_vals.append(sgn * t1_a)
                if t2_a != 0.0:
                    eq_r_idx.append(r0 + 3 + d_ax)
                    eq_c_idx.append(base + 5)
                    eq_vals.append(sgn * t2_a)
                if nv_a != 0.0:
                    eq_r_idx.append(r0 + 3 + d_ax)
                    eq_c_idx.append(base + 6)
                    eq_vals.append(sgn * nv_a)

        if ft_cap > 0.0:
            _add_ub([base + 1, eta_col], [1.0, -ft_cap])
        for su in (-1.0, 1.0):
            if sh_cap > 0.0:
                _add_ub([base + 2, base + 0, eta_col], [su, -mu, -sh_cap])
                _add_ub([base + 3, base + 0, eta_col], [su, -mu, -sh_cap])
            else:
                _add_ub([base + 2, base + 0], [su, -mu])
                _add_ub([base + 3, base + 0], [su, -mu])
        cap_m = ft_cap * m_arm
        for st in (-1.0, 1.0):
            if cap_m > 0.0:
                _add_ub([base + 4, base + 0, eta_col], [st, -hbv, -cap_m])
                _add_ub([base + 5, base + 0, eta_col], [st, -hbu, -cap_m])
            else:
                _add_ub([base + 4, base + 0], [st, -hbv])
                _add_ub([base + 5, base + 0], [st, -hbu])
            if tn_cap > 0.0:
                _add_ub([base + 6, base + 0, eta_col], [st, -0.5 * mu * m_arm, -tn_cap])
            else:
                _add_ub([base + 6, base + 0], [st, -0.5 * mu * m_arm])

    bounds.append((0.0, None))
    n_ub = len(ub_rhs)
    b_ub = np.asarray(ub_rhs, dtype=np.float64)

    c_obj = np.zeros(n_vars, dtype=np.float64)
    c_obj[eta_col] = 1.0 + 1e-6 * float(n_p)
    a_eq_csc = csc_matrix((np.asarray(eq_vals, dtype=np.float64), (eq_r_idx, eq_c_idx)), shape=(n_eq, n_vars))
    a_ub_csc = csc_matrix((np.asarray(ub_vals, dtype=np.float64), (ub_r_idx, ub_c_idx)), shape=(n_ub, n_vars))
    tol_ok = 2e-4 if n_vars > 1200 else 1e-5
    lb_arr = np.asarray([0.0 if b[0] is None else float(b[0]) for b in bounds], dtype=np.float64)
    ub_arr = np.asarray([float("inf") if b[1] is None else float(b[1]) for b in bounds], dtype=np.float64)
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
                options={"run_crossover": "off", "ipm_optimality_tolerance": 1e-8},
            )
        if res is not None and res.x is not None:
            x_ipm = np.asarray(res.x, dtype=np.float64)
            if x_ipm.shape == (n_vars,) and np.all(np.isfinite(x_ipm)):
                eta_ipm = float(x_ipm[eta_col])
                eq_ipm = float(np.max(np.abs(a_eq_csc.dot(x_ipm) - b_eq))) if n_eq > 0 else 0.0
                ub_ipm = float(np.max(np.maximum(a_ub_csc.dot(x_ipm) - b_ub, 0.0))) if n_ub > 0 else 0.0
                lb_ipm = float(np.max(np.maximum(lb_arr - x_ipm, 0.0)))
                hi_ipm = float(np.max(np.where(np.isfinite(ub_arr), np.maximum(x_ipm - ub_arr, 0.0), 0.0)))
                if (
                    eta_ipm <= 1.0 + tol_ok
                    and eq_ipm <= tol_ok
                    and ub_ipm <= tol_ok
                    and lb_ipm <= tol_ok
                    and hi_ipm <= tol_ok
                ):
                    return True, eta_ipm, eq_ipm, ub_ipm
    if res is None or not res.success or res.x is None:
        res = linprog(
            c_obj, A_ub=a_ub_csc, b_ub=b_ub, A_eq=a_eq_csc, b_eq=b_eq, bounds=bounds, method="highs"
        )
    if not res.success or res.x is None:
        return False, float("inf"), float("inf"), float("inf")

    x_sol = np.asarray(res.x, dtype=np.float64)
    eta_val = float(x_sol[eta_col])
    eq_res = float(np.max(np.abs(a_eq_csc.dot(x_sol) - b_eq))) if n_eq > 0 else 0.0
    ub_viol = float(np.max(np.maximum(a_ub_csc.dot(x_sol) - b_ub, 0.0))) if n_ub > 0 else 0.0
    is_ok = (eta_val <= 1.0 + tol_ok) and (eq_res <= tol_ok) and (ub_viol <= tol_ok)
    if not is_ok and n_vars > 1200 and eta_val <= 1.0 + 1e-3:
        res_sx = linprog(
            c_obj, A_ub=a_ub_csc, b_ub=b_ub, A_eq=a_eq_csc, b_eq=b_eq, bounds=bounds, method="highs"
        )
        if res_sx.success and res_sx.x is not None:
            x_sol = np.asarray(res_sx.x, dtype=np.float64)
            eta_val = float(x_sol[eta_col])
            eq_res = float(np.max(np.abs(a_eq_csc.dot(x_sol) - b_eq))) if n_eq > 0 else 0.0
            ub_viol = float(np.max(np.maximum(a_ub_csc.dot(x_sol) - b_ub, 0.0))) if n_ub > 0 else 0.0
            is_ok = (eta_val <= 1.0 + tol_ok) and (eq_res <= tol_ok) and (ub_viol <= tol_ok)
    return is_ok, eta_val, eq_res, ub_viol


def _try_independent_plumb_prefix_witness(
    parts_dict: Dict[int, Dict[str, Any]],
    world_boxes: Dict[int, Tuple[LDUBox, ...]],
    foundation_policy: str,
    joint_assumptions: Dict[str, float],
    gravity_unit: Tuple[float, float, float],
    ground_plane_z_ldu: float = 0.0,
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
    if gravity_unit != (0.0, 0.0, -1.0):
        return False, float("inf")
    gz_ref = float(ground_plane_z_ldu)
    is_table_only = foundation_policy == "table_friction_only"
    if is_table_only:
        # Verify every grounded brick's own CoM lies strictly inside its own z=ground_plane_z_ldu table contact rectangle
        for gid, ge in parts_dict.items():
            if bool(ge["is_grounded"]):
                gbxs = world_boxes[gid]
                tv_g = sum(float(b.volume_ldu3) for b in gbxs)
                g_cx = sum(0.5 * (b.xmin + b.xmax) * float(b.volume_ldu3) for b in gbxs) / max(tv_g, 1.0)
                g_cy = sum(0.5 * (b.ymin + b.ymax) * float(b.volume_ldu3) for b in gbxs) / max(tv_g, 1.0)
                if not any(
                    abs(float(b.zmin) - gz_ref) <= 0.25
                    and (b.xmin + 0.5) <= g_cx <= (b.xmax - 0.5)
                    and (b.ymin + 0.5) <= g_cy <= (b.ymax - 0.5)
                    for b in gbxs
                ):
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

    def _try_top_down_two_pier_bearing() -> Tuple[bool, float]:
        """Exact zero-tension top-down 1-or-2-pier bearing compression witness (`0.0` force & torque residual)."""
        f_acc: Dict[int, float] = {}
        wx_acc: Dict[int, float] = {}
        wy_acc: Dict[int, float] = {}
        for k_id, k_e in parts_dict.items():
            m_k = float(get_brick_primitive(str(k_e["part_id"])).mass_grams)
            kbxs = world_boxes[k_id]
            kv = sum(float(b.volume_ldu3) for b in kbxs)
            kcx = sum(0.5 * (b.xmin + b.xmax) * float(b.volume_ldu3) for b in kbxs) / max(kv, 1.0)
            kcy = sum(0.5 * (b.ymin + b.ymax) * float(b.volume_ldu3) for b in kbxs) / max(kv, 1.0)
            f_acc[k_id] = m_k
            wx_acc[k_id] = m_k * kcx
            wy_acc[k_id] = m_k * kcy

        top_down_ids = sorted(
            parts_dict.keys(),
            key=lambda i: (-min(b.zmin for b in world_boxes[i]), -max(b.zmax for b in world_boxes[i]), -i),
        )
        max_comp_u = 0.0
        comp_cap_per_stud = 2.5 * clutch_f
        for u_id in top_down_ids:
            ft = f_acc[u_id]
            qx = wx_acc[u_id] / max(ft, 1e-12)
            qy = wy_acc[u_id] / max(ft, 1e-12)
            if bool(parts_dict[u_id]["is_grounded"]):
                if is_table_only:
                    if not any(
                        abs(float(b.zmin) - gz_ref) <= 0.25
                        and (b.xmin + 0.5) <= qx <= (b.xmax - 0.5)
                        and (b.ymin + 0.5) <= qy <= (b.ymax - 0.5)
                        for b in world_boxes[u_id]
                    ):
                        return False, float("inf")
                else:
                    ubxs = world_boxes[u_id]
                    ux0, ux1 = min(b.xmin for b in ubxs), max(b.xmax for b in ubxs)
                    uy0, uy1 = min(b.ymin for b in ubxs), max(b.ymax for b in ubxs)
                    if not ((ux0 - 4.0) <= qx <= (ux1 + 4.0) and (uy0 - 4.0) <= qy <= (uy1 + 4.0)):
                        return False, float("inf")
                continue

            cands_u = lower_neighbors.get(u_id, ())
            if not cands_u:
                return False, float("inf")
            m_cap_1 = clutch_f * m_arm
            routed_1 = False
            for lid, pts in cands_u:
                k_l = float(len(pts))
                rx_l = sum(p[0] for p in pts) / k_l
                ry_l = sum(p[1] for p in pts) / k_l
                px0, px1 = min(p[0] for p in pts) - 7.5, max(p[0] for p in pts) + 7.5
                py0, py1 = min(p[1] for p in pts) - 7.5, max(p[1] for p in pts) + 7.5
                u_c = ft / max(1e-9, k_l * comp_cap_per_stud)
                if px0 <= qx <= px1 and py0 <= qy <= py1 and u_c <= 0.90:
                    u_eff = u_c
                else:
                    hbx = max(8.0, 0.5 * (max(p[0] for p in pts) - min(p[0] for p in pts)))
                    hby = max(8.0, 0.5 * (max(p[1] for p in pts) - min(p[1] for p in pts)))
                    u_eff = max(
                        u_c,
                        max(0.0, (abs(qy - ry_l) - hby) * ft) / max(1e-9, k_l * m_cap_1),
                        max(0.0, (abs(qx - rx_l) - hbx) * ft) / max(1e-9, k_l * m_cap_1),
                    )
                if u_eff <= 0.85:
                    f_acc[lid] += ft
                    wx_acc[lid] += ft * qx
                    wy_acc[lid] += ft * qy
                    if u_eff > max_comp_u:
                        max_comp_u = u_eff
                    routed_1 = True
                    break
            if routed_1:
                continue

            routed_2 = False
            n_cu = len(cands_u)
            for ia in range(n_cu):
                lid_a, pts_a = cands_u[ia]
                ax = sum(p[0] for p in pts_a) / float(len(pts_a))
                ay = sum(p[1] for p in pts_a) / float(len(pts_a))
                for ib in range(ia + 1, n_cu):
                    lid_b, pts_b = cands_u[ib]
                    bx = sum(p[0] for p in pts_b) / float(len(pts_b))
                    by = sum(p[1] for p in pts_b) / float(len(pts_b))
                    abx, aby = bx - ax, by - ay
                    denom = abx * abx + aby * aby
                    if denom < 1e-6:
                        continue
                    t_seg = max(0.0, min(1.0, ((qx - ax) * abx + (qy - ay) * aby) / denom))
                    proj_x = ax + t_seg * abx
                    proj_y = ay + t_seg * aby
                    rx_res = qx - proj_x
                    ry_res = qy - proj_y
                    if abs(rx_res) <= 7.5 and abs(ry_res) <= 7.5:
                        fa = (1.0 - t_seg) * ft
                        fb = t_seg * ft
                        ua = fa / max(1e-9, float(len(pts_a)) * comp_cap_per_stud)
                        ub = fb / max(1e-9, float(len(pts_b)) * comp_cap_per_stud)
                        if max(ua, ub) <= 0.90:
                            f_acc[lid_a] += fa
                            wx_acc[lid_a] += fa * (ax + rx_res)
                            wy_acc[lid_a] += fa * (ay + ry_res)
                            f_acc[lid_b] += fb
                            wx_acc[lid_b] += fb * (bx + rx_res)
                            wy_acc[lid_b] += fb * (by + ry_res)
                            max_comp_u = max(max_comp_u, ua, ub)
                            routed_2 = True
                            break
                if routed_2:
                    break
            if not routed_2:
                return False, float("inf")
        return True, max_comp_u

    ok_2p, u_2p = _try_top_down_two_pier_bearing()
    if ok_2p:
        return True, u_2p

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

        if is_table_only:
            if not any(
                abs(float(b.zmin) - gz_ref) <= 0.25
                and (b.xmin + 1.0) <= cx <= (b.xmax - 1.0)
                and (b.ymin + 1.0) <= cy <= (b.ymax - 1.0)
                for b in world_boxes[cur_id]
            ):
                return False, float("inf")

    return True, peak_eta


def _boxes_have_continuous_half_ray_corridor(
    new_boxes: Sequence[LDUBox],
    existing_boxes: Sequence[LDUBox],
) -> bool:
    """Exact continuous 1D swept-box clearance (`t in (0, +inf)`) along any of the 6 cardinal axes (`F10`)."""
    if not new_boxes or not existing_boxes:
        return True
    if not any(
        nb.xmin < eb.xmax and nb.xmax > eb.xmin and nb.ymin < eb.ymax and nb.ymax > eb.ymin and eb.zmax > nb.zmin
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    if not any(
        nb.xmin < eb.xmax and nb.xmax > eb.xmin and nb.ymin < eb.ymax and nb.ymax > eb.ymin and eb.zmin < nb.zmax
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    if not any(
        nb.ymin < eb.ymax and nb.ymax > eb.ymin and nb.zmin < eb.zmax and nb.zmax > eb.zmin and eb.xmax > nb.xmin
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    if not any(
        nb.ymin < eb.ymax and nb.ymax > eb.ymin and nb.zmin < eb.zmax and nb.zmax > eb.zmin and eb.xmin < nb.xmax
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    if not any(
        nb.xmin < eb.xmax and nb.xmax > eb.xmin and nb.zmin < eb.zmax and nb.zmax > eb.zmin and eb.ymax > nb.ymin
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    if not any(
        nb.xmin < eb.xmax and nb.xmax > eb.xmin and nb.zmin < eb.zmax and nb.zmax > eb.zmin and eb.ymin < nb.ymax
        for nb in new_boxes
        for eb in existing_boxes
    ):
        return True
    return False


def _swept_obb_hits_half_ray(
    obb_moving: OBB3D,
    obb_static: OBB3D,
    ray_dir: np.ndarray,
    allowed_penetration_ldu: float = 0.25,
) -> bool:
    """Exact continuous Minkowski-ray SAT intersection for `obb_moving + t * ray_dir` (`t in (0, +inf)`)."""
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


def _orthonormal_tangent_se3(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(float(n[0])) < 0.85 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, ref)
    t1 /= max(float(np.linalg.norm(t1)), 1e-12)
    t2 = np.cross(n, t1)
    t2 /= max(float(np.linalg.norm(t2)), 1e-12)
    return t1, t2


def _independent_continuous_se3_equilibrium_lp(
    parts_dict: Dict[int, Dict[str, Any]],
    poses_map: Dict[int, ContinuousSE3Pose],
    obbs_map: Dict[int, Tuple[OBB3D, ...]],
    foundation_policy: str,
    joint_assumptions: Dict[str, float],
    gravity_unit: Tuple[float, float, float] = (0.0, 0.0, -1.0),
    lateral_accel_xy: Tuple[float, float] = (0.0, 0.0),
    ground_plane_z_ldu: float = 0.0,
) -> Tuple[bool, float, float, float, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Completely standalone `scipy.optimize.linprog` 6-DoF static equilibrium solver for continuous
    $SE(3)$ assemblies (`F2`: zero imports of `static_equilibrium` or `ContinuousAssemblyState`).
    Returns `(is_stable, eta, max_eq_residual_g, max_ub_violation, discovered_port_pairs)`.
    """
    if not parts_dict:
        return True, 0.0, 0.0, 0.0, []

    grounded_ids = [i for i, e in parts_dict.items() if bool(e.get("is_grounded", False))]
    if not grounded_ids:
        return False, float("inf"), float("inf"), float("inf"), []

    # Discover geometric port mates independently from `poses_map` via 20-LDU spatial cell hashing (`S5`)
    iids_sorted = sorted(parts_dict.keys())
    used_ports: Set[Tuple[int, int]] = set()
    discovered_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    cell_size = 20.0
    port_records: Dict[Tuple[int, int], Tuple[int, Any, ContinuousSE3Pose]] = {}
    spatial_buckets: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}
    for idx_ord, ia in enumerate(iids_sorted):
        pr_a = get_brick_primitive(str(parts_dict[ia]["part_id"]))
        pose_a = poses_map[ia]
        for pa in pr_a.ports:
            gpa = (ia, pa.port_id)
            wfa = pose_a.compose(ContinuousSE3Pose.from_ldu_frame(pa.local_frame))
            port_records[gpa] = (idx_ord, pa, wfa)
            ckey = (
                int(math.floor(float(wfa.t[0]) / cell_size)),
                int(math.floor(float(wfa.t[1]) / cell_size)),
                int(math.floor(float(wfa.t[2]) / cell_size)),
            )
            spatial_buckets.setdefault(ckey, []).append(gpa)

    pair_checks = 0
    for idx_a, ia in enumerate(iids_sorted):
        pr_a = get_brick_primitive(str(parts_dict[ia]["part_id"]))
        for pa in pr_a.ports:
            gpa = (ia, pa.port_id)
            if gpa in used_ports:
                continue
            _, _, wfa = port_records[gpa]
            za = wfa.R[:, 2]
            cx = int(math.floor(float(wfa.t[0]) / cell_size))
            cy = int(math.floor(float(wfa.t[1]) / cell_size))
            cz = int(math.floor(float(wfa.t[2]) / cell_size))
            best_d = 1.5
            best_gpb: Optional[Tuple[int, int]] = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nbrs = spatial_buckets.get((cx + dx, cy + dy, cz + dz))
                        if not nbrs:
                            continue
                        for gpb in nbrs:
                            if gpb in used_ports:
                                continue
                            idx_b, pb, wfb = port_records[gpb]
                            if idx_b <= idx_a:
                                continue
                            pair_checks += 1
                            if pair_checks > 500_000:
                                raise RuntimeError("se3_port_pair_budget_exceeded")
                            if not are_connectors_compatible(
                                pa.connector_type, pa.polarity, pb.connector_type, pb.polarity
                            ):
                                continue
                            dist = float(np.linalg.norm(wfa.t - wfb.t))
                            if dist > best_d or (
                                abs(dist - best_d) <= 1e-12 and best_gpb is not None and gpb >= best_gpb
                            ):
                                continue
                            is_ball = pa.connector_type in (
                                ConnectorType.BALL_JOINT,
                                ConnectorType.BALL_SOCKET,
                            )
                            is_stud_pair = pa.connector_type in (
                                ConnectorType.STANDARD_STUD,
                                ConnectorType.HOLLOW_STUD,
                            ) or pb.connector_type in (
                                ConnectorType.STANDARD_STUD,
                                ConnectorType.HOLLOW_STUD,
                            )
                            min_cos = 0.45 if is_ball else 0.96
                            cos_z = float(np.dot(za, wfb.R[:, 2]))
                            if (cos_z if is_stud_pair else abs(cos_z)) >= min_cos:
                                best_d = dist
                                best_gpb = gpb
            if best_gpb is not None:
                used_ports.add(gpa)
                used_ports.add(best_gpb)
                discovered_pairs.append((gpa, best_gpb))

    adj: Dict[int, Set[int]] = {i: set() for i in parts_dict}
    for (ia, _), (ib, _) in discovered_pairs:
        adj[ia].add(ib)
        adj[ib].add(ia)
    vis = set(grounded_ids)
    q = list(grounded_ids)
    while q:
        c = q.pop(0)
        for nb in adj[c]:
            if nb not in vis:
                vis.add(nb)
                q.append(nb)
    if len(vis) < len(parts_dict):
        return False, float("inf"), float("inf"), float("inf"), discovered_pairs

    is_table = foundation_policy == "table_friction_only"
    gz_se3 = float(ground_plane_z_ldu)
    table_pts_map: Dict[int, List[np.ndarray]] = {}
    if is_table:
        signs = np.array(
            [
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        for gid in grounded_ids:
            all_z: List[float] = []
            low_pts: List[np.ndarray] = []
            for ob in obbs_map[gid]:
                ax_m = np.asarray(ob.axes_rows, dtype=np.float64)
                he_v = np.asarray(ob.half_extents, dtype=np.float64)
                ct_v = np.asarray(ob.center, dtype=np.float64)
                crn = ct_v + (signs * he_v) @ ax_m
                for pt in crn:
                    zv = float(pt[2])
                    all_z.append(zv)
                    if abs(zv - gz_se3) <= 0.25:
                        low_pts.append(np.array([float(pt[0]), float(pt[1]), gz_se3], dtype=np.float64))
            min_zg = min(all_z) if all_z else float("inf")
            if abs(min_zg - gz_se3) > 0.25 or not low_pts:
                return False, float("inf"), float("inf"), float("inf"), discovered_pairs
            uniq_xy: Dict[Tuple[int, int], np.ndarray] = {}
            for pt in low_pts:
                uniq_xy[(int(round(pt[0] * 100.0)), int(round(pt[1] * 100.0)))] = pt
            table_pts_map[gid] = list(uniq_xy.values())

    free_insts = sorted(parts_dict.keys()) if is_table else sorted(
        i for i, e in parts_dict.items() if not bool(e.get("is_grounded", False))
    )
    if not free_insts:
        return True, 0.0, 0.0, 0.0, discovered_pairs
    free_idx_map = {iid: idx for idx, iid in enumerate(free_insts)}

    contacts: List[Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
    if is_table:
        for gid, t_pts in table_pts_map.items():
            for tk, pt_t in enumerate(t_pts):
                contacts.append(
                    (
                        (-1, -100 - tk),
                        (gid, -100 - tk),
                        pt_t,
                        np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        np.array([1.0, 0.0, 0.0], dtype=np.float64),
                        np.array([0.0, 1.0, 0.0], dtype=np.float64),
                        "table_vertex",
                    )
                )

    for gpa, gpb in discovered_pairs:
        ia, pa = gpa
        ib, pb = gpb
        if (ia not in free_idx_map) and (ib not in free_idx_map):
            continue
        sa = get_brick_primitive(str(parts_dict[ia]["part_id"])).ports[pa]
        sb = get_brick_primitive(str(parts_dict[ib]["part_id"])).ports[pb]
        wfa = poses_map[ia].compose(ContinuousSE3Pose.from_ldu_frame(sa.local_frame))
        p_c = wfa.t
        if sa.polarity == PortPolarity.STUD:
            n_c = wfa.R[:, 2]
            ord_a, ord_b = gpa, gpb
        else:
            wfb = poses_map[ib].compose(ContinuousSE3Pose.from_ldu_frame(sb.local_frame))
            n_c = wfb.R[:, 2]
            ord_a, ord_b = gpb, gpa
        t1, t2 = _orthonormal_tangent_se3(n_c)
        names = {sa.connector_type.name, sb.connector_type.name}
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
        contacts.append((ord_a, ord_b, p_c, n_c, t1, t2, jmode))

    n_free = len(free_insts)
    n_c = len(contacts)
    if n_c == 0:
        return False, float("inf"), float("inf"), float("inf"), discovered_pairs
    num_vars = 12 * n_c + 1
    eta_col = 12 * n_c

    g_norm = math.sqrt(sum(float(c) * float(c) for c in gravity_unit))
    gx_u, gy_u, gz_u = (
        (float(gravity_unit[0]) / g_norm, float(gravity_unit[1]) / g_norm, float(gravity_unit[2]) / g_norm)
        if g_norm > 1e-12
        else (0.0, 0.0, -1.0)
    )

    eq_rows: List[int] = []
    eq_cols: List[int] = []
    eq_vals: List[float] = []
    b_eq = np.zeros(6 * n_free, dtype=np.float64)
    coms: Dict[int, np.ndarray] = {}
    for iid in free_insts:
        pr = get_brick_primitive(str(parts_dict[iid]["part_id"]))
        b_idx = free_idx_map[iid]
        m_g = float(pr.mass_grams)
        b_eq[6 * b_idx + 0] = (-gx_u + float(lateral_accel_xy[0])) * m_g
        b_eq[6 * b_idx + 1] = (-gy_u + float(lateral_accel_xy[1])) * m_g
        b_eq[6 * b_idx + 2] = -gz_u * m_g
        vol_s = 0.0
        lc = np.zeros(3, dtype=np.float64)
        for bx in pr.hull_boxes:
            v = float(bx.volume_ldu3)
            vol_s += v
            lc += v * np.array([0.5 * (bx.xmin + bx.xmax), 0.5 * (bx.ymin + bx.ymax), 0.5 * (bx.zmin + bx.zmax)])
        lc /= max(vol_s, 1.0)
        coms[iid] = poses_map[iid].R @ lc + poses_map[iid].t

    bounds_list: List[Tuple[float, Optional[float]]] = [(0.0, None)] * num_vars
    ub_rows: List[int] = []
    ub_cols: List[int] = []
    ub_vals: List[float] = []
    b_ub_list: List[float] = []

    f_max = float(joint_assumptions.get("clutch_force_capacity_grams", 99.9))
    crush_ratio = float(joint_assumptions.get("normal_crush_ratio", 25.0))
    f_comp_max = f_max * crush_ratio
    m_arm = float(joint_assumptions.get("moment_arm_capacity_ldu", 8.0))
    m_max = f_max * m_arm
    mu = float(joint_assumptions.get("coulomb_friction_coeff", 0.60))
    detent_m = float(joint_assumptions.get("locking_hinge_detent_moment_ldu_g", 380.0))
    ball_m = float(joint_assumptions.get("ball_joint_friction_moment_ldu_g", 220.0))

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

        if jmode == "table_vertex":
            bounds_list[base + 5] = (0.0, 0.0)
            for k_z in range(6, 12):
                bounds_list[base + k_z] = (0.0, 0.0)
            r0 = len(b_ub_list)
            ub_rows.extend((r0, r0))
            ub_cols.extend((base + 4, eta_col))
            ub_vals.extend((1.0, -f_comp_max))
            b_ub_list.append(0.0)
            r2 = len(b_ub_list)
            ub_rows.extend([r2] * 5)
            ub_cols.extend((base, base + 1, base + 2, base + 3, base + 4))
            ub_vals.extend((1.0, 1.0, 1.0, 1.0, -mu))
            b_ub_list.append(0.0)
            continue

        if jmode == "smooth_hinge":
            bounds_list[base + 10] = (0.0, 0.0)
            bounds_list[base + 11] = (0.0, 0.0)
        elif jmode == "axle":
            bounds_list[base + 5] = (0.0, 0.0)

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

        r2 = len(b_ub_list)
        ub_rows.extend([r2] * 7)
        ub_cols.extend((base, base + 1, base + 2, base + 3, base + 4, base + 5, eta_col))
        ub_vals.extend((1.0, 1.0, 1.0, 1.0, -mu, +mu, -f_max))
        b_ub_list.append(0.0)

        if jmode == "ball_joint":
            r3 = len(b_ub_list)
            ub_rows.extend([r3] * 7)
            ub_cols.extend((base + 6, base + 7, base + 8, base + 9, base + 10, base + 11, eta_col))
            ub_vals.extend((1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -ball_m))
            b_ub_list.append(0.0)
        elif jmode == "locking_hinge":
            r3 = len(b_ub_list)
            ub_rows.extend([r3] * 5)
            ub_cols.extend((base + 6, base + 7, base + 8, base + 9, eta_col))
            ub_vals.extend((1.0, 1.0, 1.0, 1.0, -m_max))
            b_ub_list.append(0.0)
            r4 = len(b_ub_list)
            ub_rows.extend((r4, r4, r4))
            ub_cols.extend((base + 10, base + 11, eta_col))
            ub_vals.extend((1.0, 1.0, -detent_m))
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
    c_obj = np.zeros(num_vars, dtype=np.float64)
    c_obj[eta_col] = 1.0
    c_obj[:eta_col] = 1e-7

    res = linprog(c_obj, A_ub=a_ub_csc, b_ub=b_ub, A_eq=a_eq_csc, b_eq=b_eq, bounds=bounds_list, method="highs")
    if not res.success or res.x is None:
        return False, float("inf"), float("inf"), float("inf"), discovered_pairs
    sol = np.asarray(res.x, dtype=np.float64)
    eta = float(sol[eta_col])
    eq_res_g = float(np.max(np.abs(a_eq_csc @ sol - b_eq))) if b_eq.size > 0 else 0.0
    ub_viol = float(max(0.0, float(np.max(a_ub_csc @ sol - b_ub)))) if b_ub.size > 0 else 0.0
    return (eta <= 1.0 + 1e-6 and eq_res_g <= 1e-4 and ub_viol <= 1e-4), eta, eq_res_g, ub_viol, discovered_pairs


class VerifierBudgetExceededError(RuntimeError):
    """Raised when an untrusted certificate exceeds configured resource or wall-time budgets (`S5`)."""


@dataclass(frozen=True, slots=True)
class VerifierResourceLimits:
    """Bounded resource ceilings for certificate verification workers (`Phase 13 S5`)."""

    max_json_bytes: int = 32 * 1024 * 1024
    max_placed_parts: int = 5_000
    max_execution_ops: int = 15_000
    max_subtrace_ops: int = 5_000
    max_matched_port_pairs: int = 50_000
    max_forbidden_boxes: int = 1_000
    wall_time_budget_sec: Optional[float] = 60.0


@dataclass(frozen=True, slots=True)
class MechanicsVerificationProfile:
    """
    Relying-party mechanics & tolerance bounds (`Phase 13 S4`):
    validates physical parameter ranges on every certificate and optionally overrides
    submitter-supplied `joint_assumptions` / `tolerances` with relying-party policy.
    """

    allowed_foundation_policies: Optional[FrozenSet[str]] = None
    max_clutch_force_capacity_grams: float = 250.0
    max_moment_arm_capacity_ldu: float = 80.0
    max_friction_coeff: float = 1.0
    max_coulomb_friction_coeff: float = 1.0
    max_normal_crush_ratio: float = 600.0
    max_locking_hinge_detent_moment_ldu_g: float = 1000.0
    max_ball_joint_friction_moment_ldu_g: float = 1000.0
    max_eq_residual_tol_g: float = 2e-4
    max_ub_violation_tol: float = 2e-4
    override_joint_assumptions: Optional[Mapping[str, float]] = None
    override_tolerances: Optional[Mapping[str, float]] = None


@dataclass(frozen=True, slots=True)
class VerificationObligations:
    """
    Caller-specified verification obligations (`Phase 12 F1 & F7` + `Phase 13 S3, S4, S5, S8`):
    decouples verification requirements, target overlap, inventory, and mechanics profile from
    untrusted `artifact` claims.
    """

    require_static_mechanics: bool = True
    require_prefix_mechanics: bool = True
    require_monotone_insertion: bool = True
    require_inventory: bool = True
    require_perturbation_if_articulated: bool = False
    require_loop_closure_if_cyclic: bool = False
    expected_target_field: Optional[Any] = None
    expected_target_sha256: Optional[str] = None
    expected_forbidden_boxes: Tuple[LDUBox, ...] = ()
    expected_license_filter: Optional[str] = None
    min_target_iou: Optional[float] = None
    require_explicit_inventory_spec: bool = False
    expected_inventory_counts: Optional[Mapping[str, int]] = None
    expected_colored_inventory: Optional[Mapping[str, int]] = None
    expected_allowed_part_ids: Optional[FrozenSet[str]] = None
    expected_foundation_policy: Optional[str] = None
    mechanics_profile: Optional[MechanicsVerificationProfile] = None
    trusted_custom_catalog_sha256: Optional[str] = None
    resource_limits: Optional[VerifierResourceLimits] = None
    expected_ground_plane_z_ldu: Optional[float] = 0.0


@dataclass(frozen=True, slots=True)
class IndependentVerificationReport:
    """Result returned by `verify_certificate_independently(artifact)` (`R9-WS1` §3.1 + `Phase 12/13`)."""

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
    verification_scope: str = "assembly_only"
    target_verified: Optional[bool] = None
    target_digest_verified: Optional[bool] = None
    target_overlap_verified: Optional[bool] = None
    measured_target_iou: Optional[float] = None
    inventory_status: str = "unchecked_absent"
    mechanics_profile_sha256: str = ""


def _compute_mechanics_profile_sha256(
    foundation_policy: str,
    joint_assumptions: Mapping[str, Any],
    tolerances: Mapping[str, Any],
    ground_plane_z_ldu: float = 0.0,
) -> str:
    payload = {
        "foundation_policy": str(foundation_policy),
        "ground_plane_z_ldu": round(float(ground_plane_z_ldu), 6),
        "joint_assumptions": {
            str(k): round(float(v), 8)
            for k, v in sorted(joint_assumptions.items())
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        },
        "tolerances": {
            str(k): round(float(v), 10)
            for k, v in sorted(tolerances.items())
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_and_resolve_mechanics_inputs(
    artifact: VersionedCertificateArtifact,
    profile: MechanicsVerificationProfile,
) -> Tuple[bool, Dict[str, float], Dict[str, float], List[str]]:
    """Validate finite physical ranges and apply relying-party overrides (`S4`)."""
    errs: List[str] = []
    raw_j: Dict[str, Any] = dict(artifact.joint_assumptions or {})
    if profile.override_joint_assumptions is not None:
        raw_j.update(dict(profile.override_joint_assumptions))

    raw_t: Dict[str, Any] = dict(artifact.tolerances or {})
    if profile.override_tolerances is not None:
        raw_t.update(dict(profile.override_tolerances))

    joint_bounds: Dict[str, Tuple[float, float, bool]] = {
        # (lo, hi, strictly_positive)
        "clutch_force_capacity_grams": (0.0, profile.max_clutch_force_capacity_grams, True),
        "moment_arm_capacity_ldu": (0.0, profile.max_moment_arm_capacity_ldu, True),
        "friction_coeff": (0.0, profile.max_friction_coeff, False),
        "coulomb_friction_coeff": (0.0, profile.max_coulomb_friction_coeff, False),
        "normal_crush_ratio": (1.0, profile.max_normal_crush_ratio, False),
        "locking_hinge_detent_moment_ldu_g": (0.0, profile.max_locking_hinge_detent_moment_ldu_g, False),
        "ball_joint_friction_moment_ldu_g": (0.0, profile.max_ball_joint_friction_moment_ldu_g, False),
    }
    eff_j: Dict[str, float] = {}
    for k, v in raw_j.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append(f"invalid_mechanics_parameter:{k}:non_numeric")
            continue
        fv = float(v)
        if not math.isfinite(fv):
            errs.append(f"invalid_mechanics_parameter:{k}:non_finite")
            continue
        if k in joint_bounds:
            lo, hi, strict_pos = joint_bounds[k]
            if (strict_pos and fv <= lo) or (not strict_pos and fv < lo) or fv > hi:
                errs.append(f"invalid_mechanics_parameter:{k}={fv}:outside_[{lo},{hi}]")
                continue
        eff_j[str(k)] = fv

    tol_bounds: Dict[str, Tuple[float, float]] = {
        "eq_residual_tol_g": (0.0, profile.max_eq_residual_tol_g),
        "ub_violation_tol": (0.0, profile.max_ub_violation_tol),
        "fine_csg_voxel_ldu": (0.25, 4.0),
    }
    eff_t: Dict[str, float] = {}
    for k, v in raw_t.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append(f"invalid_tolerance_parameter:{k}:non_numeric")
            continue
        fv = float(v)
        if not math.isfinite(fv):
            errs.append(f"invalid_tolerance_parameter:{k}:non_finite")
            continue
        if k in tol_bounds:
            lo, hi = tol_bounds[k]
            if fv < lo or fv > hi:
                errs.append(f"invalid_tolerance_parameter:{k}={fv}:outside_[{lo},{hi}]")
                continue
        eff_t[str(k)] = fv

    return (len(errs) == 0), eff_j, eff_t, errs


def verify_certificate_independently(
    artifact: VersionedCertificateArtifact,
    obligations: Optional[VerificationObligations] = None,
) -> IndependentVerificationReport:
    """
    Independently replay and verify a `VersionedCertificateArtifact` with ZERO `TopoBrickKernel`
    or `static_equilibrium` imports (`F1, F2, F7, F10` + `S3, S4, S5, S8`).
    """
    obl = obligations if obligations is not None else VerificationObligations()
    try:
        return _verify_certificate_independently_impl(artifact, obl)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        has_t = (
            obl.expected_target_field is not None
            or obl.expected_target_sha256 is not None
            or obl.min_target_iou is not None
        )
        return IndependentVerificationReport(
            verified=False,
            schema_version_ok=False,
            catalog_sha256_valid=False,
            inventory_valid=False,
            csg_2ldu_collision_free=False,
            ports_compatible=False,
            monotone_insertion_valid=False,
            static_mechanics_valid=False,
            prefix_mechanics_valid=False,
            witness_residuals_valid=False,
            max_eq_residual_g=float("inf"),
            max_ub_violation=float("inf"),
            replayed_peak_utilization=float("inf"),
            failure_reasons=(f"verifier_exception:{type(exc).__name__}:{exc}",),
            verification_scope="assembly_and_target" if has_t else "assembly_only",
            target_verified=False if has_t else None,
            target_digest_verified=False if (obl.expected_target_field is not None or obl.expected_target_sha256 is not None) else None,
            target_overlap_verified=False if obl.min_target_iou is not None else None,
            measured_target_iou=None,
            inventory_status="failed",
            mechanics_profile_sha256="",
        )


def _verify_certificate_independently_impl(
    artifact: VersionedCertificateArtifact,
    obl: VerificationObligations,
) -> IndependentVerificationReport:
    limits = obl.resource_limits if obl.resource_limits is not None else VerifierResourceLimits()
    prof = obl.mechanics_profile if obl.mechanics_profile is not None else MechanicsVerificationProfile()
    t_deadline: Optional[float] = (
        time.monotonic() + float(limits.wall_time_budget_sec)
        if limits.wall_time_budget_sec is not None and float(limits.wall_time_budget_sec) > 0.0
        else None
    )

    def _check_deadline() -> None:
        if t_deadline is not None and time.monotonic() > t_deadline:
            raise VerifierBudgetExceededError(
                f"wall_time_budget_exceeded:{limits.wall_time_budget_sec}s"
            )

    reasons: List[str] = []
    has_target_request = (
        obl.expected_target_field is not None
        or obl.expected_target_sha256 is not None
        or obl.min_target_iou is not None
    )
    ver_scope = "assembly_and_target" if has_target_request else "assembly_only"
    target_digest_ver: Optional[bool] = None
    target_overlap_ver: Optional[bool] = None
    measured_iou: Optional[float] = None

    # Validate target-overlap obligations upfront BEFORE replay (`S3`: never silently skip min_target_iou)
    min_iou_threshold: Optional[float] = None
    if obl.min_target_iou is not None:
        try:
            min_iou_threshold = float(obl.min_target_iou)
        except (TypeError, ValueError):
            min_iou_threshold = float("nan")
        if (
            isinstance(obl.min_target_iou, bool)
            or not math.isfinite(min_iou_threshold)
            or not (0.0 <= min_iou_threshold <= 1.0)
        ):
            target_overlap_ver = False
            reasons.append("invalid_obligation:min_target_iou_out_of_range")
            min_iou_threshold = None
        elif obl.expected_target_field is None or not callable(
            getattr(obl.expected_target_field, "intersection_volume", None)
        ):
            target_overlap_ver = False
            reasons.append("invalid_obligation:min_target_iou_requires_expected_target_field")
        elif artifact.assembly_mode != "cubic_ldu":
            target_overlap_ver = False
            reasons.append(
                f"unsupported_obligation:min_target_iou_unsupported_for_assembly_mode:{artifact.assembly_mode}"
            )

    def _finalize_target_verdict() -> Optional[bool]:
        if not has_target_request:
            return None
        digest_needed = (obl.expected_target_field is not None) or (obl.expected_target_sha256 is not None)
        overlap_needed = obl.min_target_iou is not None
        ok_d = (not digest_needed) or (target_digest_ver is True)
        ok_o = (not overlap_needed) or (target_overlap_ver is True)
        return bool(ok_d and ok_o)

    # Check resource bounds upfront (`S5`)
    if len(artifact.placed_parts) > limits.max_placed_parts:
        reasons.append(f"resource_limit_exceeded:placed_parts>{limits.max_placed_parts}")
    if len(artifact.execution_operations) > limits.max_execution_ops:
        reasons.append(f"resource_limit_exceeded:execution_operations>{limits.max_execution_ops}")
    if len(artifact.matched_port_pairs) > limits.max_matched_port_pairs:
        reasons.append(f"resource_limit_exceeded:matched_port_pairs>{limits.max_matched_port_pairs}")
    if len(artifact.forbidden_support_boxes) + len(obl.expected_forbidden_boxes) > limits.max_forbidden_boxes:
        reasons.append(f"resource_limit_exceeded:forbidden_support_boxes>{limits.max_forbidden_boxes}")

    schema_ok = artifact.schema_version in SUPPORTED_CERTIFICATE_SCHEMAS
    if not schema_ok:
        reasons.append(f"unsupported_schema:{artifact.schema_version}")

    eff_schema_for_sha = (
        artifact.schema_version
        if artifact.schema_version in SUPPORTED_CERTIFICATE_SCHEMAS
        else CERTIFICATE_SCHEMA_VERSION
    )
    try:
        expected_sha = compute_catalog_sha256(
            schema_version=eff_schema_for_sha,
            recompute_from_disk=True,
        )
        cat_ok = hmac.compare_digest(str(artifact.catalog_sha256).strip().lower(), expected_sha)
        if not cat_ok:
            reasons.append("catalog_sha256_mismatch")
        # Enforce release-bundle provenance unless custom digest explicitly trusted (`S8`)
        if obl.trusted_custom_catalog_sha256 is not None:
            if not is_valid_sha256_hex(obl.trusted_custom_catalog_sha256):
                cat_ok = False
                reasons.append("invalid_obligation:trusted_custom_catalog_sha256")
            elif not hmac.compare_digest(
                expected_sha, obl.trusted_custom_catalog_sha256.strip().lower()
            ):
                cat_ok = False
                reasons.append("trusted_custom_catalog_sha256_mismatch")
        else:
            from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

            if COMPILED_LDRAW_CATALOG.is_available:
                pinned_rel_sha = PINNED_RELEASE_CATALOG_SHA256_BY_SCHEMA.get(eff_schema_for_sha)
                if pinned_rel_sha is not None and not hmac.compare_digest(expected_sha, pinned_rel_sha):
                    cat_ok = False
                    reasons.append("local_catalog_differs_from_pinned_release")
    except ValueError as exc_cat:
        cat_ok = False
        reasons.append(f"catalog_archive_tampered:{exc_cat}")

    mode_ok = artifact.assembly_mode in _VALID_ASSEMBLY_MODES
    if not mode_ok:
        reasons.append(f"unsupported_assembly_mode:{artifact.assembly_mode}")

    eff_foundation_policy = str(artifact.foundation_policy)
    fp_ok = eff_foundation_policy in _VALID_FOUNDATION_POLICIES
    if not fp_ok:
        reasons.append(f"unsupported_foundation_policy:{eff_foundation_policy}")
    if obl.expected_foundation_policy is not None and eff_foundation_policy != obl.expected_foundation_policy:
        fp_ok = False
        reasons.append(
            f"foundation_policy_mismatch:expected={obl.expected_foundation_policy},got={eff_foundation_policy}"
        )
    if (
        prof.allowed_foundation_policies is not None
        and eff_foundation_policy not in prof.allowed_foundation_policies
    ):
        fp_ok = False
        reasons.append(f"disallowed_foundation_policy:{eff_foundation_policy}")

    eff_ground_z = float(getattr(artifact, "ground_plane_z_ldu", 0.0))
    if not math.isfinite(eff_ground_z) or abs(eff_ground_z) > 1e6:
        fp_ok = False
        reasons.append(f"invalid_ground_plane_z_ldu:{eff_ground_z}")
    elif obl.expected_ground_plane_z_ldu is not None:
        if (
            isinstance(obl.expected_ground_plane_z_ldu, bool)
            or not isinstance(obl.expected_ground_plane_z_ldu, (int, float))
            or not math.isfinite(float(obl.expected_ground_plane_z_ldu))
        ):
            fp_ok = False
            reasons.append("invalid_obligation:expected_ground_plane_z_ldu")
        elif abs(eff_ground_z - float(obl.expected_ground_plane_z_ldu)) > 1e-4:
            fp_ok = False
            reasons.append(
                f"ground_plane_z_ldu_mismatch:expected={float(obl.expected_ground_plane_z_ldu)},got={eff_ground_z}"
            )

    mech_params_ok, eff_joint_assumptions, eff_tolerances, mech_errs = _validate_and_resolve_mechanics_inputs(
        artifact, prof
    )
    if not mech_params_ok:
        reasons.extend(mech_errs)
    mech_profile_sha = _compute_mechanics_profile_sha256(
        eff_foundation_policy,
        eff_joint_assumptions,
        eff_tolerances,
        ground_plane_z_ldu=(eff_ground_z if math.isfinite(eff_ground_z) else 0.0),
    )

    if any(r.startswith("resource_limit_exceeded:") for r in reasons):
        return IndependentVerificationReport(
            verified=False,
            schema_version_ok=schema_ok,
            catalog_sha256_valid=cat_ok,
            inventory_valid=False,
            csg_2ldu_collision_free=False,
            ports_compatible=False,
            monotone_insertion_valid=False,
            static_mechanics_valid=False,
            prefix_mechanics_valid=False,
            witness_residuals_valid=False,
            max_eq_residual_g=float("inf"),
            max_ub_violation=float("inf"),
            replayed_peak_utilization=float("inf"),
            failure_reasons=tuple(reasons),
            verification_scope=ver_scope,
            target_verified=False if has_target_request else None,
            target_digest_verified=target_digest_ver,
            target_overlap_verified=target_overlap_ver,
            measured_target_iou=None,
            inventory_status="failed",
            mechanics_profile_sha256=mech_profile_sha,
        )

    # Validate placed_parts upfront without uncaught exceptions
    parts_by_id: Dict[int, Dict[str, Any]] = {}
    placed_valid = True
    for raw_e in artifact.placed_parts:
        _check_deadline()
        try:
            iid_e = int(raw_e["instance_id"])
            pid_e = str(raw_e["part_id"])
            pos_raw = raw_e["pos"]
            if len(pos_raw) != 3:
                raise ValueError("pos_len")
            if iid_e in parts_by_id:
                placed_valid = False
                reasons.append(f"duplicate_placed_instance_id:{iid_e}")
                continue
            if not is_known_part_id(pid_e):
                placed_valid = False
                reasons.append(f"unknown_part_id:{pid_e}")
                continue
            if artifact.assembly_mode == "cubic_ldu":
                rid_e = int(raw_e["rot_id"])
                if not (0 <= rid_e < 24):
                    raise ValueError("bad_rot_id")
                _ = (int(pos_raw[0]), int(pos_raw[1]), int(pos_raw[2]))
            elif artifact.assembly_mode == "continuous_se3":
                rmat = np.asarray(raw_e["rot_matrix"], dtype=np.float64)
                pvec = np.asarray(pos_raw, dtype=np.float64)
                if rmat.shape != (3, 3) or pvec.shape != (3,) or not np.all(np.isfinite(rmat)) or not np.all(np.isfinite(pvec)):
                    raise ValueError("bad_se3_pose")
                orth_err = float(np.max(np.abs(rmat.T @ rmat - np.eye(3))))
                det_r = float(np.linalg.det(rmat))
                if orth_err > 1e-3 or abs(det_r - 1.0) > 1e-3:
                    raise ValueError("non_so3_rot_matrix")
            parts_by_id[iid_e] = raw_e
        except (KeyError, TypeError, ValueError) as exc_p:
            placed_valid = False
            reasons.append(f"malformed_placed_part:{exc_p}")

    # Check claimed_certificates against active VerificationObligations (`F1`)
    claims = artifact.claimed_certificates
    has_parts = len(artifact.placed_parts) > 0
    if claims.get("geometry") != "PASS":
        reasons.append(f"claimed_certificate_not_pass:geometry={claims.get('geometry')}")
    if obl.require_static_mechanics and claims.get("static_mechanics") != "PASS":
        reasons.append(f"claimed_certificate_not_pass:static_mechanics={claims.get('static_mechanics')}")
    if obl.require_prefix_mechanics:
        allowed_pref = ("PASS",) if has_parts else ("PASS", "NOT_APPLICABLE")
        if claims.get("prefix_mechanics") not in allowed_pref:
            reasons.append(f"claimed_certificate_not_pass:prefix_mechanics={claims.get('prefix_mechanics')}")
    if obl.require_monotone_insertion:
        allowed_ins = ("PASS",) if has_parts else ("PASS", "NOT_APPLICABLE")
        if claims.get("insertion") not in allowed_ins:
            reasons.append(f"claimed_certificate_not_pass:insertion={claims.get('insertion')}")
    if obl.require_inventory and (artifact.inventory_spec is not None or artifact.colored_inventory_spec is not None):
        if "inventory" in claims and claims.get("inventory") not in ("PASS",):
            reasons.append(f"claimed_certificate_not_pass:inventory={claims.get('inventory')}")

    inv_ok = True
    has_relying_party_inv = (
        obl.expected_inventory_counts is not None
        or obl.expected_colored_inventory is not None
        or obl.expected_allowed_part_ids is not None
    )
    has_submitter_inv = (artifact.inventory_spec is not None) or (artifact.colored_inventory_spec is not None)

    if obl.require_explicit_inventory_spec and not has_relying_party_inv and not has_submitter_inv:
        inv_ok = False
        reasons.append("missing_inventory_spec")

    if placed_valid and obl.expected_allowed_part_ids is not None:
        allowed_norm = {str(x).strip() for x in obl.expected_allowed_part_ids} | {
            str(x).strip().lower() for x in obl.expected_allowed_part_ids
        }
        for entry in artifact.placed_parts:
            pid_a = str(entry["part_id"])
            if pid_a not in allowed_norm and pid_a.lower() not in allowed_norm:
                inv_ok = False
                reasons.append(f"disallowed_part_id:{pid_a}")
                break

    def _check_mold_counts(cap_map: Mapping[str, int], tag: str) -> None:
        nonlocal inv_ok
        counts_m: Dict[str, int] = {}
        for entry in artifact.placed_parts:
            pid = str(entry["part_id"])
            p_key = pid if pid in EXTENDED_BRICK_CATALOG else pid.lower()
            counts_m[p_key] = counts_m.get(p_key, 0) + 1
        for pid, used in counts_m.items():
            cap_m = int(cap_map.get(pid, cap_map.get(pid.lower(), 0)))
            if used > cap_m:
                inv_ok = False
                reasons.append(f"{tag}:{pid}")
                break

    def _check_colored_counts(col_map: Mapping[str, int], tag: str) -> None:
        nonlocal inv_ok
        col_counts_m: Dict[str, int] = {}
        for entry in artifact.placed_parts:
            pid = str(entry["part_id"])
            p_key = pid if pid in EXTENDED_BRICK_CATALOG else pid.lower()
            cid = int(entry.get("color_id", 15))
            ckey = f"{p_key}:{cid}"
            col_counts_m[ckey] = col_counts_m.get(ckey, 0) + 1
        for ckey, used_c in col_counts_m.items():
            raw_pid, raw_cid = ckey.rsplit(":", 1)
            cap_c = int(col_map.get(ckey, col_map.get(f"{raw_pid.lower()}:{raw_cid}", 0)))
            if used_c > cap_c:
                inv_ok = False
                reasons.append(f"{tag}:{ckey}")
                break

    if inv_ok and placed_valid and obl.expected_inventory_counts is not None:
        _check_mold_counts(obl.expected_inventory_counts, "expected_inventory_exceeded")
    if inv_ok and placed_valid and obl.expected_colored_inventory is not None:
        _check_colored_counts(obl.expected_colored_inventory, "expected_colored_inventory_exceeded")

    if inv_ok and obl.require_inventory and artifact.inventory_spec is not None and placed_valid:
        _check_mold_counts(artifact.inventory_spec, "inventory_exceeded")
    if inv_ok and obl.require_inventory and artifact.colored_inventory_spec is not None and placed_valid:
        _check_colored_counts(artifact.colored_inventory_spec, "colored_inventory_exceeded")

    # License filter check (`F7`)
    eff_lic = obl.expected_license_filter if obl.expected_license_filter is not None else artifact.license_filter
    if eff_lic not in ("all", "cc_by_4_0_only"):
        inv_ok = False
        reasons.append(f"unsupported_license_filter:{eff_lic}")
    elif eff_lic == "cc_by_4_0_only" and placed_valid:
        from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

        COMPILED_LDRAW_CATALOG._ensure_loaded()
        for entry in artifact.placed_parts:
            pid_l = str(entry["part_id"])
            if pid_l in EXTENDED_BRICK_CATALOG or pid_l.lower() in EXTENDED_BRICK_CATALOG:
                continue
            r_id = COMPILED_LDRAW_CATALOG.resolve_id(pid_l)
            rec_l = COMPILED_LDRAW_CATALOG._raw_parts.get(r_id or pid_l.lower(), {})
            if str(rec_l.get("t", "")).startswith("TIER_B"):
                inv_ok = False
                reasons.append(f"license_filter_violated:{pid_l}")
                break

    inv_status = (
        "failed"
        if not inv_ok
        else (
            "verified_relying_party"
            if has_relying_party_inv
            else ("verified_submitter_spec" if (obl.require_inventory and has_submitter_inv) else "unchecked_absent")
        )
    )

    # Target digest & overlap measurement (`S3`: separate digest agreement vs. geometric fit)
    if obl.expected_target_sha256 is not None or obl.expected_target_field is not None:
        exp_t_sha = (
            obl.expected_target_sha256
            if obl.expected_target_sha256 is not None
            else compute_target_sha256(obl.expected_target_field)
        )
        target_digest_ver = artifact.target_sha256 == exp_t_sha
        if not target_digest_ver:
            reasons.append("target_sha256_mismatch")

    if (
        obl.expected_target_field is not None
        and callable(getattr(obl.expected_target_field, "intersection_volume", None))
        and artifact.assembly_mode == "cubic_ldu"
        and placed_valid
    ):
        all_wb = [
            b.transform(LDUFrame(int(e_t["rot_id"]), tuple(int(c) for c in e_t["pos"])))
            for e_t in artifact.placed_parts
            for b in get_brick_primitive(str(e_t["part_id"])).hull_boxes
        ]
        inter_v = float(obl.expected_target_field.intersection_volume(all_wb))
        t_v = float(getattr(obl.expected_target_field, "total_volume_ldu3", 0))
        asm_v = float(sum(b.volume_ldu3 for b in all_wb))
        union_v = max(t_v + asm_v - inter_v, 1.0)
        measured_iou = inter_v / union_v if (t_v > 0 or asm_v > 0) else 1.0
        if min_iou_threshold is not None and target_overlap_ver is None:
            target_overlap_ver = bool(measured_iou >= min_iou_threshold - 1e-6)
            if not target_overlap_ver:
                reasons.append(f"target_iou_below_minimum:{measured_iou:.4f}<{min_iou_threshold:.4f}")

    target_ver = _finalize_target_verdict()

    if not mode_ok or not fp_ok or not placed_valid or not mech_params_ok:
        return IndependentVerificationReport(
            verified=False,
            schema_version_ok=schema_ok,
            catalog_sha256_valid=cat_ok,
            inventory_valid=inv_ok,
            csg_2ldu_collision_free=False,
            ports_compatible=False,
            monotone_insertion_valid=False,
            static_mechanics_valid=False,
            prefix_mechanics_valid=False,
            witness_residuals_valid=False,
            max_eq_residual_g=float("inf"),
            max_ub_violation=float("inf"),
            replayed_peak_utilization=float("inf"),
            failure_reasons=tuple(reasons),
            verification_scope=ver_scope,
            target_verified=target_ver,
            target_digest_verified=target_digest_ver,
            target_overlap_verified=target_overlap_ver,
            measured_target_iou=measured_iou,
            inventory_status=inv_status,
            mechanics_profile_sha256=mech_profile_sha,
        )

    _check_deadline()
    csg_ok, csg_pairs = check_fine_csg_voxel_collision_2ldu(artifact.placed_parts)
    if not csg_ok:
        reasons.append(f"csg_2ldu_collision:{csg_pairs}")

    # Forbidden boxes check (`F7`)
    forb_boxes: List[LDUBox] = [
        LDUBox(int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3]), int(fb[4]), int(fb[5]))
        for fb in artifact.forbidden_support_boxes
    ] + list(obl.expected_forbidden_boxes)
    if forb_boxes and artifact.assembly_mode == "cubic_ldu":
        for e_fb in artifact.placed_parts:
            pr_fb = get_brick_primitive(str(e_fb["part_id"]))
            fr_fb = LDUFrame(int(e_fb["rot_id"]), tuple(int(c) for c in e_fb["pos"]))
            if any(b.transform(fr_fb).intersects(fbox) for b in pr_fb.hull_boxes for fbox in forb_boxes):
                csg_ok = False
                reasons.append(f"forbidden_support_box_violated:{e_fb['instance_id']}")
                break
    elif forb_boxes and artifact.assembly_mode == "continuous_se3":
        for e_fb in artifact.placed_parts:
            pr_fb = get_brick_primitive(str(e_fb["part_id"]))
            ps_fb = ContinuousSE3Pose.from_matrix_and_pos(
                np.asarray(e_fb["rot_matrix"], dtype=np.float64),
                tuple(float(x) for x in e_fb["pos"]),
            )
            if any(
                obb_intersects_sat(box_to_obb(b, ps_fb), box_to_obb(fbox, ContinuousSE3Pose.identity()), allowed_penetration_ldu=0.05)
                for b in pr_fb.hull_boxes
                for fbox in forb_boxes
            ):
                csg_ok = False
                reasons.append(f"forbidden_support_box_violated:{e_fb['instance_id']}")
                break

    # Port index & connector compatibility check (`F1`: zero uncaught IndexError)
    ports_ok = True
    for (ia, pa_id), (ib, pb_id) in artifact.matched_port_pairs:
        if ia not in parts_by_id or ib not in parts_by_id or ia == ib:
            ports_ok = False
            reasons.append("invalid_matched_port_instance")
            break
        pr_a_m = get_brick_primitive(str(parts_by_id[ia]["part_id"]))
        pr_b_m = get_brick_primitive(str(parts_by_id[ib]["part_id"]))
        if not (0 <= int(pa_id) < len(pr_a_m.ports)) or not (0 <= int(pb_id) < len(pr_b_m.ports)):
            ports_ok = False
            reasons.append(f"invalid_port_index:({ia},{pa_id})<->({ib},{pb_id})")
            break
        sa = pr_a_m.ports[int(pa_id)]
        sb = pr_b_m.ports[int(pb_id)]
        if not are_connectors_compatible(sa.connector_type, sa.polarity, sb.connector_type, sb.polarity):
            ports_ok = False
            reasons.append("incompatible_matched_ports")
            break

    # Witness summary residual check against validated tolerances (`S4`)
    eq_tol = float(eff_tolerances.get("eq_residual_tol_g", 1e-5))
    ub_tol = float(eff_tolerances.get("ub_violation_tol", 1e-5))
    w_eq = float((artifact.lp_witness_summary or {}).get("max_eq_residual_g", 0.0))
    w_ub = float((artifact.lp_witness_summary or {}).get("max_ub_violation", 0.0))
    witness_res_ok = (
        math.isfinite(w_eq)
        and math.isfinite(w_ub)
        and (0.0 <= w_eq <= eq_tol)
        and (0.0 <= w_ub <= ub_tol)
    )
    if not witness_res_ok:
        reasons.append(f"witness_residual_exceeded:eq={w_eq:.2e},ub={w_ub:.2e}")

    need_trace = has_parts and (obl.require_prefix_mechanics or obl.require_monotone_insertion)
    ins_ok = True
    prefix_ok = True
    static_ok = True
    max_eq_r = w_eq
    max_ub_v = w_ub
    final_eta = 0.0

    if need_trace and len(artifact.execution_operations) == 0:
        if obl.require_monotone_insertion:
            ins_ok = False
        if obl.require_prefix_mechanics:
            prefix_ok = False
        reasons.append("missing_execution_trace")

    if artifact.assembly_mode == "cubic_ldu":
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
            return _boxes_have_continuous_half_ray_corridor(new_boxes, existing_boxes)

        def _collides_with_sim(new_boxes: Sequence[LDUBox]) -> bool:
            return any(
                nb.intersects(eb)
                for bxs_e in sim_boxes.values()
                for eb in bxs_e
                for nb in new_boxes
            )

        m_arm_v = float(eff_joint_assumptions.get("moment_arm_capacity_ldu", 8.0))
        is_table_fp = eff_foundation_policy == "table_friction_only"
        inc_port_grid: Dict[Tuple[int, int, int], List[Tuple[int, PortPolarity]]] = {}
        inc_lower_nbrs: Dict[int, Dict[int, List[Tuple[float, float, float]]]] = {}
        bearing_prefix_fails = 0

        def _register_inc_part(p_iid: int, p_entry: Dict[str, Any], p_bxs: Tuple[LDUBox, ...]) -> None:
            pr_i = get_brick_primitive(str(p_entry["part_id"]))
            fr_i = LDUFrame(int(p_entry["rot_id"]), tuple(int(c) for c in p_entry["pos"]))
            zmin_i = min(b.zmin for b in p_bxs)
            zmax_i = max(b.zmax for b in p_bxs)
            for p_s in pr_i.ports:
                if p_s.connector_type not in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
                    continue
                wf_i = fr_i.compose(p_s.local_frame)
                if abs(wf_i.z_axis[2]) != 1:
                    continue
                existing_at_pos = inc_port_grid.setdefault(wf_i.pos, [])
                pt_f = (float(wf_i.pos[0]), float(wf_i.pos[1]), float(wf_i.pos[2]))
                for o_iid, o_pol in existing_at_pos:
                    if o_iid == p_iid or o_pol == p_s.polarity or o_iid not in sim_boxes:
                        continue
                    o_bxs = sim_boxes[o_iid]
                    zmin_o = min(b.zmin for b in o_bxs)
                    zmax_o = max(b.zmax for b in o_bxs)
                    if zmax_o <= zmin_i:
                        inc_lower_nbrs.setdefault(p_iid, {}).setdefault(o_iid, []).append(pt_f)
                    elif zmax_i <= zmin_o:
                        inc_lower_nbrs.setdefault(o_iid, {}).setdefault(p_iid, []).append(pt_f)
                existing_at_pos.append((p_iid, p_s.polarity))

        def _unregister_inc_part(p_iid: int) -> None:
            inc_lower_nbrs.pop(p_iid, None)
            for d_low in inc_lower_nbrs.values():
                d_low.pop(p_iid, None)
            for wpos_k in list(inc_port_grid.keys()):
                filt = [it for it in inc_port_grid[wpos_k] if it[0] != p_iid]
                if filt:
                    inc_port_grid[wpos_k] = filt
                else:
                    inc_port_grid.pop(wpos_k, None)

        def _ground_ray_ok(g_iid: int, qx: float, qy: float) -> bool:
            gbxs = sim_boxes[g_iid]
            if is_table_fp:
                return any(
                    abs(float(b.zmin) - eff_ground_z) <= 0.25
                    and (b.xmin + 0.5) <= qx <= (b.xmax - 0.5)
                    and (b.ymin + 0.5) <= qy <= (b.ymax - 0.5)
                    for b in gbxs
                )
            gx0 = float(min(b.xmin for b in gbxs))
            gx1 = float(max(b.xmax for b in gbxs))
            gy0 = float(min(b.ymin for b in gbxs))
            gy1 = float(max(b.ymax for b in gbxs))
            return (gx0 + 0.25) <= qx <= (gx1 - 0.25) and (gy0 + 0.25) <= qy <= (gy1 - 0.25)

        def _patch_kernel_ok(u_iid: int, l_iid: int, pts: List[Tuple[float, float, float]], qx: float, qy: float) -> bool:
            rx_p = sum(p[0] for p in pts) / float(len(pts))
            ry_p = sum(p[1] for p in pts) / float(len(pts))
            ubxs = sim_boxes[u_iid]
            lbxs = sim_boxes[l_iid]
            ox0 = float(max(min(b.xmin for b in ubxs), min(b.xmin for b in lbxs)))
            ox1 = float(min(max(b.xmax for b in ubxs), max(b.xmax for b in lbxs)))
            oy0 = float(max(min(b.ymin for b in ubxs), min(b.ymin for b in lbxs)))
            oy1 = float(min(max(b.ymax for b in ubxs), max(b.ymax for b in lbxs)))
            hb_x = max(min(rx_p - ox0, ox1 - rx_p) if ox0 < rx_p < ox1 else 0.0, m_arm_v)
            hb_y = max(min(ry_p - oy0, oy1 - ry_p) if oy0 < ry_p < oy1 else 0.0, m_arm_v)
            return abs(qx - rx_p) <= (hb_x - 0.25) and abs(qy - ry_p) <= (hb_y - 0.25)

        def _greedy_single_interior_ray(cur_u: int, qx: float, qy: float, vis_init: Set[int]) -> bool:
            vis_s = set(vis_init)
            while not bool(sim_parts[cur_u].get("is_grounded", False)):
                low_map_s = inc_lower_nbrs.get(cur_u)
                if not low_map_s:
                    return False
                cands_s = [(lid, pts) for lid, pts in low_map_s.items() if lid not in vis_s and lid in sim_parts]
                if not cands_s:
                    return False
                cands_s.sort(
                    key=lambda item: (
                        (qx - sum(p[0] for p in item[1]) / float(len(item[1]))) ** 2
                        + (qy - sum(p[1] for p in item[1]) / float(len(item[1]))) ** 2,
                        item[0],
                    )
                )
                picked = None
                for lid_s, pts_s in cands_s:
                    if not _patch_kernel_ok(cur_u, lid_s, pts_s, qx, qy):
                        continue
                    if bool(sim_parts[lid_s].get("is_grounded", False)):
                        if _ground_ray_ok(lid_s, qx, qy):
                            return True
                        continue
                    picked = lid_s
                    break
                if picked is None:
                    return False
                vis_s.add(picked)
                cur_u = picked
            return _ground_ray_ok(cur_u, qx, qy)

        def _can_extend_prefix_interior_ray(
            cur_u: int, qx: float, qy: float, splits_left: int = 2, vis_set: Optional[Set[int]] = None
        ) -> bool:
            vis = set(vis_set) if vis_set is not None else {cur_u}
            if _greedy_single_interior_ray(cur_u, qx, qy, vis):
                return True
            if splits_left <= 0:
                return False
            while not bool(sim_parts[cur_u].get("is_grounded", False)):
                low_map = inc_lower_nbrs.get(cur_u)
                if not low_map:
                    return False
                cands = [(lid, pts) for lid, pts in low_map.items() if lid not in vis and lid in sim_parts]
                if not cands:
                    return False
                cands.sort(
                    key=lambda item: (
                        (qx - sum(p[0] for p in item[1]) / float(len(item[1]))) ** 2
                        + (qy - sum(p[1] for p in item[1]) / float(len(item[1]))) ** 2,
                        item[0],
                    )
                )
                if len(cands) >= 2:
                    n_c = len(cands)
                    for ia in range(n_c):
                        la, pa_pts = cands[ia]
                        ax = sum(p[0] for p in pa_pts) / float(len(pa_pts))
                        ay = sum(p[1] for p in pa_pts) / float(len(pa_pts))
                        for ib in range(ia + 1, n_c):
                            lb, pb_pts = cands[ib]
                            bx = sum(p[0] for p in pb_pts) / float(len(pb_pts))
                            by = sum(p[1] for p in pb_pts) / float(len(pb_pts))
                            abx, aby = bx - ax, by - ay
                            denom = abx * abx + aby * aby
                            if denom < 1e-6:
                                continue
                            t_s = max(0.05, min(0.95, ((qx - ax) * abx + (qy - ay) * aby) / denom))
                            rx_r = qx - (ax + t_s * abx)
                            ry_r = qy - (ay + t_s * aby)
                            if abs(rx_r) > 7.5 or abs(ry_r) > 7.5:
                                continue
                            qax, qay = ax + rx_r, ay + ry_r
                            qbx, qby = bx + rx_r, by + ry_r
                            if not _patch_kernel_ok(cur_u, la, pa_pts, qax, qay):
                                continue
                            if not _patch_kernel_ok(cur_u, lb, pb_pts, qbx, qby):
                                continue
                            ok_a = (
                                _ground_ray_ok(la, qax, qay)
                                if bool(sim_parts[la].get("is_grounded", False))
                                else _can_extend_prefix_interior_ray(la, qax, qay, splits_left - 1, vis | {la})
                            )
                            if not ok_a:
                                continue
                            ok_b = (
                                _ground_ray_ok(lb, qbx, qby)
                                if bool(sim_parts[lb].get("is_grounded", False))
                                else _can_extend_prefix_interior_ray(lb, qbx, qby, splits_left - 1, vis | {lb})
                            )
                            if ok_b:
                                return True
                nxt_single = None
                for lid, pts in cands:
                    if not _patch_kernel_ok(cur_u, lid, pts, qx, qy):
                        continue
                    if not bool(sim_parts[lid].get("is_grounded", False)):
                        nxt_single = lid
                        break
                if nxt_single is None:
                    return False
                vis.add(nxt_single)
                cur_u = nxt_single
            return False

        if (obl.require_prefix_mechanics or obl.require_monotone_insertion) and len(artifact.execution_operations) > 0:
            total_subtrace_ops = 0
            for op in artifact.execution_operations:
                _check_deadline()
                op_t = str(op.get("op_type", ""))
                added_only_grounded = False
                just_placed_iid: Optional[int] = None
                if op_t == "PLACE_PART":
                    iids = op.get("instance_ids", [])
                    if len(iids) != 1:
                        prefix_ok = False
                        reasons.append("malformed_trace_op:PLACE_PART_instance_ids")
                        break
                    iid = int(iids[0])
                    if iid in sim_parts:
                        prefix_ok = False
                        reasons.append(f"duplicate_trace_instance:{iid}")
                        break
                    raw_pid_op = str(op.get("part_id", "")).strip()
                    if raw_pid_op:
                        if not is_known_part_id(raw_pid_op):
                            prefix_ok = False
                            reasons.append(f"unknown_trace_part_id:{raw_pid_op}")
                            break
                        try:
                            rid_op = int(op.get("rot_id", 0))
                            pos_op = [int(c) for c in op["pos"]]
                            if not (0 <= rid_op < 24) or len(pos_op) != 3:
                                raise ValueError("bad_cubic_op_pose")
                        except (KeyError, TypeError, ValueError):
                            prefix_ok = False
                            reasons.append("malformed_trace_op:PLACE_PART_pose")
                            break
                        pe = {
                            "instance_id": iid,
                            "part_id": raw_pid_op,
                            "rot_id": rid_op,
                            "pos": pos_op,
                            "is_grounded": bool(op.get("is_grounded", False)),
                            "color_id": int(
                                op.get(
                                    "color_id",
                                    parts_by_id[iid].get("color_id", 15) if iid in parts_by_id else 15,
                                )
                            ),
                        }
                    elif iid in parts_by_id:
                        pe = dict(parts_by_id[iid])
                    else:
                        prefix_ok = False
                        reasons.append(f"missing_trace_part_spec:{iid}")
                        break
                    bxs_pe = _boxes_of(pe)
                    if _collides_with_sim(bxs_pe):
                        csg_ok = False
                        prefix_ok = False
                        reasons.append(f"trace_step_collision:{iid}")
                        break
                    if obl.require_monotone_insertion and not _has_corridor(bxs_pe):
                        ins_ok = False
                    sim_parts[iid] = pe
                    sim_boxes[iid] = bxs_pe
                    _register_inc_part(iid, pe, bxs_pe)
                    added_only_grounded = bool(pe.get("is_grounded", False))
                    just_placed_iid = iid
                elif op_t == "PLACE_SUBASSEMBLY":
                    exp_sub_iids = [int(x) for x in op.get("instance_ids", [])]
                    if not exp_sub_iids or any(sid in sim_parts or sid not in parts_by_id for sid in exp_sub_iids):
                        prefix_ok = False
                        reasons.append("malformed_subassembly_instance_ids")
                        break
                    sub_ops = op.get("subtrace") or []
                    if sub_ops:
                        total_subtrace_ops += len(sub_ops)
                        if total_subtrace_ops > limits.max_subtrace_ops:
                            raise VerifierBudgetExceededError(
                                f"subtrace_ops>{limits.max_subtrace_ops}"
                            )
                        loc_sim_parts: Dict[int, Dict[str, Any]] = {}
                        loc_sim_boxes: Dict[int, Tuple[LDUBox, ...]] = {}
                        loc_prev_stable = True
                        for idx_sub, s_op in enumerate(sub_ops):
                            if str(s_op.get("op_type", "")) != "PLACE_PART":
                                prefix_ok = False
                                reasons.append(f"unknown_subtrace_op_type:{s_op.get('op_type')}")
                                break
                            s_iids = s_op.get("instance_ids", [])
                            s_iid = int(s_iids[0]) if s_iids else (idx_sub + 1)
                            if s_iid in loc_sim_parts:
                                prefix_ok = False
                                reasons.append(f"duplicate_subtrace_instance:{s_iid}")
                                break
                            s_pid = str(s_op.get("part_id", "")).strip()
                            if not is_known_part_id(s_pid):
                                prefix_ok = False
                                reasons.append(f"unknown_subtrace_part_id:{s_pid}")
                                break
                            s_pe = {
                                "instance_id": s_iid,
                                "part_id": s_pid,
                                "rot_id": int(s_op.get("rot_id", 0)),
                                "pos": [int(c) for c in s_op["pos"]],
                                "is_grounded": bool(s_op.get("is_grounded", False)),
                            }
                            s_bxs = _boxes_of(s_pe)
                            loc_existing = [b for lb in loc_sim_boxes.values() for b in lb]
                            if any(nb.intersects(eb) for eb in loc_existing for nb in s_bxs):
                                csg_ok = False
                                prefix_ok = False
                                reasons.append(f"subtrace_step_collision:{s_iid}")
                                break
                            if obl.require_monotone_insertion and not _boxes_have_continuous_half_ray_corridor(
                                s_bxs, loc_existing
                            ):
                                ins_ok = False
                            loc_sim_parts[s_iid] = s_pe
                            loc_sim_boxes[s_iid] = s_bxs
                            if obl.require_prefix_mechanics:
                                if bool(s_pe["is_grounded"]) and loc_prev_stable:
                                    continue
                                if len(loc_sim_parts) >= 20:
                                    p_ok_l, _ = _try_independent_plumb_prefix_witness(
                                        loc_sim_parts,
                                        loc_sim_boxes,
                                        "clutched_baseplate",
                                        eff_joint_assumptions,
                                        (0.0, 0.0, -1.0),
                                    )
                                    if p_ok_l:
                                        loc_prev_stable = True
                                        continue
                                sub_st_ok, _, sub_eq_r, sub_ub_v = _independent_cubic_equilibrium_lp(
                                    loc_sim_parts,
                                    "clutched_baseplate",
                                    eff_joint_assumptions,
                                )
                                max_eq_r = max(max_eq_r, sub_eq_r)
                                max_ub_v = max(max_ub_v, sub_ub_v)
                                if not sub_st_ok:
                                    prefix_ok = False
                                    loc_prev_stable = False
                                    break
                                loc_prev_stable = True
                        if not prefix_ok:
                            reasons.append(f"subtrace_prefix_unstable:{op.get('part_id', '')}")
                            break
                        if loc_sim_parts:
                            sub_csg_ok, sub_csg_pairs = check_fine_csg_voxel_collision_2ldu(tuple(loc_sim_parts.values()))
                            if not sub_csg_ok:
                                csg_ok = False
                                reasons.append(f"subtrace_csg_2ldu_collision:{sub_csg_pairs}")
                        if len(loc_sim_parts) != len(exp_sub_iids):
                            prefix_ok = False
                            reasons.append("subtrace_instance_count_mismatch")
                            break
                        inferred_tf: Optional[LDUFrame] = None
                        for check_sid in exp_sub_iids:
                            if check_sid not in loc_sim_parts or check_sid not in parts_by_id:
                                prefix_ok = False
                                reasons.append(f"subtrace_missing_instance:{check_sid}")
                                break
                            lp_e = loc_sim_parts[check_sid]
                            wp_e = parts_by_id[check_sid]
                            if str(lp_e["part_id"]) != str(wp_e["part_id"]):
                                prefix_ok = False
                                reasons.append(f"subtrace_pose_or_mold_mismatch:{check_sid}")
                                break
                            lp_fr = LDUFrame(int(lp_e["rot_id"]), tuple(int(c) for c in lp_e["pos"]))
                            wp_fr = LDUFrame(int(wp_e["rot_id"]), tuple(int(c) for c in wp_e["pos"]))
                            cur_tf = wp_fr.compose(lp_fr.inverse())
                            if inferred_tf is None:
                                inferred_tf = cur_tf
                            elif cur_tf != inferred_tf:
                                prefix_ok = False
                                reasons.append(f"subtrace_non_rigid_offset:{check_sid}")
                                break
                        if not prefix_ok:
                            break

                    grp_boxes: List[LDUBox] = []
                    for iid_g in exp_sub_iids:
                        grp_boxes.extend(_boxes_of(parts_by_id[iid_g]))
                    if _collides_with_sim(grp_boxes):
                        csg_ok = False
                        prefix_ok = False
                        reasons.append("trace_subassembly_collision")
                        break
                    if obl.require_monotone_insertion and not _has_corridor(grp_boxes):
                        ins_ok = False
                    for iid_g in exp_sub_iids:
                        pe_g = dict(parts_by_id[iid_g])
                        bxs_g = _boxes_of(pe_g)
                        sim_parts[iid_g] = pe_g
                        sim_boxes[iid_g] = bxs_g
                        _register_inc_part(iid_g, pe_g, bxs_g)
                elif op_t == "ADD_TEMP_SUPPORT":
                    tag_t = str(op.get("temp_support_tag", "")).strip()
                    pid_t = str(op.get("part_id", "")).strip()
                    if not tag_t or tag_t in temp_tags or not is_known_part_id(pid_t):
                        prefix_ok = False
                        reasons.append(f"malformed_temp_support:{tag_t}")
                        break
                    tid = next_tmp_id
                    next_tmp_id += 1
                    temp_tags[tag_t] = tid
                    pe_t = {
                        "instance_id": tid,
                        "part_id": pid_t,
                        "rot_id": int(op.get("rot_id", 0)),
                        "pos": [int(c) for c in op["pos"]],
                        "is_grounded": bool(op.get("is_grounded", False)),
                    }
                    bxs_t = _boxes_of(pe_t)
                    if _collides_with_sim(bxs_t):
                        csg_ok = False
                        prefix_ok = False
                        reasons.append(f"temp_support_collision:{tag_t}")
                        break
                    sim_parts[tid] = pe_t
                    sim_boxes[tid] = bxs_t
                    _register_inc_part(tid, pe_t, bxs_t)
                elif op_t == "REMOVE_TEMP_SUPPORT":
                    tag_rem = str(op.get("temp_support_tag", "")).strip()
                    if tag_rem not in temp_tags:
                        prefix_ok = False
                        reasons.append(f"unknown_temp_support_tag:{tag_rem}")
                        break
                    tid_rem = temp_tags.pop(tag_rem)
                    _unregister_inc_part(tid_rem)
                    sim_parts.pop(tid_rem, None)
                    sim_boxes.pop(tid_rem, None)
                elif op_t == "REORIENT_ASSEMBLY":
                    gu = op.get("gravity_unit", [0.0, 0.0, -1.0])
                    if len(gu) != 3 or math.sqrt(sum(float(x) * float(x) for x in gu)) < 1e-6:
                        prefix_ok = False
                        reasons.append("invalid_reorient_gravity")
                        break
                    cur_grav = (float(gu[0]), float(gu[1]), float(gu[2]))
                else:
                    prefix_ok = False
                    reasons.append(f"unknown_op_type:{op_t}")
                    break

                if obl.require_prefix_mechanics and sim_parts:
                    if prev_prefix_stable and cur_grav == (0.0, 0.0, -1.0):
                        if added_only_grounded and just_placed_iid is not None:
                            bxs_j = sim_boxes[just_placed_iid]
                            tv_j = sum(float(b.volume_ldu3) for b in bxs_j)
                            j_cx = sum(0.5 * (b.xmin + b.xmax) * float(b.volume_ldu3) for b in bxs_j) / max(tv_j, 1.0)
                            j_cy = sum(0.5 * (b.ymin + b.ymax) * float(b.volume_ldu3) for b in bxs_j) / max(tv_j, 1.0)
                            if not is_table_fp or _ground_ray_ok(just_placed_iid, j_cx, j_cy):
                                continue
                        elif just_placed_iid is not None and len(sim_parts) >= 12:
                            bxs_j = sim_boxes[just_placed_iid]
                            tv_j = sum(float(b.volume_ldu3) for b in bxs_j)
                            j_cx = sum(0.5 * (b.xmin + b.xmax) * float(b.volume_ldu3) for b in bxs_j) / max(tv_j, 1.0)
                            j_cy = sum(0.5 * (b.ymin + b.ymax) * float(b.volume_ldu3) for b in bxs_j) / max(tv_j, 1.0)
                            if _can_extend_prefix_interior_ray(just_placed_iid, j_cx, j_cy, 3):
                                continue
                    if len(sim_parts) >= 24 and bearing_prefix_fails < 8:
                        plumb_ok, _ = _try_independent_plumb_prefix_witness(
                            sim_parts,
                            sim_boxes,
                            eff_foundation_policy,
                            eff_joint_assumptions,
                            cur_grav,
                            ground_plane_z_ldu=eff_ground_z,
                        )
                        if plumb_ok:
                            bearing_prefix_fails = 0
                            prev_prefix_stable = True
                            continue
                        bearing_prefix_fails += 1
                    st_step, _, eq_r_s, ub_v_s = _independent_cubic_equilibrium_lp(
                        sim_parts,
                        eff_foundation_policy,
                        eff_joint_assumptions,
                        gravity_unit=cur_grav,
                        ground_plane_z_ldu=eff_ground_z,
                    )
                    max_eq_r = max(max_eq_r, eq_r_s)
                    max_ub_v = max(max_ub_v, ub_v_s)
                    if not st_step:
                        prefix_ok = False
                        prev_prefix_stable = False
                        break
                    prev_prefix_stable = True

            # Verify terminal trace state matches `placed_parts` identically (`F1`)
            if prefix_ok and ins_ok:
                if temp_tags:
                    prefix_ok = False
                    reasons.append(f"unremoved_temp_support:{sorted(temp_tags.keys())}")
                elif set(sim_parts.keys()) != set(parts_by_id.keys()):
                    prefix_ok = False
                    reasons.append("trace_terminal_instance_mismatch")
                else:
                    for iid_chk, w_e in parts_by_id.items():
                        s_e = sim_parts[iid_chk]
                        if (
                            str(s_e["part_id"]) != str(w_e["part_id"])
                            or int(s_e["rot_id"]) != int(w_e["rot_id"])
                            or [int(c) for c in s_e["pos"]] != [int(c) for c in w_e["pos"]]
                            or bool(s_e.get("is_grounded", False)) != bool(w_e.get("is_grounded", False))
                            or int(s_e.get("color_id", 15)) != int(w_e.get("color_id", 15))
                        ):
                            prefix_ok = False
                            reasons.append(f"trace_terminal_state_mismatch:{iid_chk}")
                            break

        if obl.require_static_mechanics:
            _check_deadline()
            static_ok, final_eta, eq_r_f, ub_v_f = _independent_cubic_equilibrium_lp(
                parts_by_id,
                eff_foundation_policy,
                eff_joint_assumptions,
                ground_plane_z_ldu=eff_ground_z,
            )
            max_eq_r = max(max_eq_r, eq_r_f)
            max_ub_v = max(max_ub_v, ub_v_f)
    else:
        # Standalone Continuous SE(3) verification (`F2`: 0 imports of ContinuousAssemblyState or static_equilibrium!)
        poses_all: Dict[int, ContinuousSE3Pose] = {}
        obbs_all: Dict[int, Tuple[OBB3D, ...]] = {}
        for iid_c, entry_c in parts_by_id.items():
            pose_c = ContinuousSE3Pose.from_matrix_and_pos(
                np.asarray(entry_c["rot_matrix"], dtype=np.float64),
                tuple(float(x) for x in entry_c["pos"]),
            )
            pr_c = get_brick_primitive(str(entry_c["part_id"]))
            poses_all[iid_c] = pose_c
            obbs_all[iid_c] = tuple(box_to_obb(b, pose_c) for b in pr_c.hull_boxes)

        # Standalone 15-axis OBB-SAT + socket-pocket protrusion check across `parts_by_id`
        c_iids = sorted(parts_by_id.keys())
        matched_lookup: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for ga_m, gb_m in artifact.matched_port_pairs:
            matched_lookup[ga_m] = gb_m
            matched_lookup[gb_m] = ga_m
        for idx_a in range(len(c_iids)):
            ia_c = c_iids[idx_a]
            for idx_b in range(idx_a + 1, len(c_iids)):
                ib_c = c_iids[idx_b]
                if any(
                    obb_intersects_sat(oa, ob, allowed_penetration_ldu=0.25)
                    for oa in obbs_all[ia_c]
                    for ob in obbs_all[ib_c]
                ):
                    csg_ok = False
                    reasons.append(f"se3_obb_hull_collision:({ia_c},{ib_c})")
                    break
            if not csg_ok:
                break

        if csg_ok:
            for ia_c in c_iids:
                pr_ca = get_brick_primitive(str(parts_by_id[ia_c]["part_id"]))
                for ps_c in pr_ca.ports:
                    if ps_c.local_protrusion_box is None:
                        continue
                    wfa_p = poses_all[ia_c].compose(ContinuousSE3Pose.from_ldu_frame(ps_c.local_frame))
                    prot_o = box_to_obb(ps_c.local_protrusion_box, wfa_p)
                    part_m = matched_lookup.get((ia_c, ps_c.port_id))
                    p_iid = part_m[0] if part_m is not None else -1
                    p_wf: Optional[ContinuousSE3Pose] = None
                    p_ct: Optional[ConnectorType] = None
                    if part_m is not None and p_iid in parts_by_id:
                        ps_b_m = get_brick_primitive(str(parts_by_id[p_iid]["part_id"])).ports[part_m[1]]
                        p_wf = poses_all[p_iid].compose(ContinuousSE3Pose.from_ldu_frame(ps_b_m.local_frame))
                        p_ct = ps_b_m.connector_type
                    for ib_c in c_iids:
                        if ib_c == ia_c:
                            continue
                        for ob_b in obbs_all[ib_c]:
                            if (
                                ib_c == p_iid
                                and p_wf is not None
                                and p_ct is not None
                                and is_obb_protrusion_contained_in_socket_pocket(prot_o, ob_b, p_wf, p_ct)
                            ):
                                continue
                            if obb_intersects_sat(prot_o, ob_b, allowed_penetration_ldu=0.25):
                                csg_ok = False
                                reasons.append(f"se3_protrusion_collision:({ia_c},{ib_c})")
                                break
                        if not csg_ok:
                            break
                    if not csg_ok:
                        break
                if not csg_ok:
                    break

        # Replay continuous trace step-by-step
        if (obl.require_prefix_mechanics or obl.require_monotone_insertion) and len(artifact.execution_operations) > 0:
            sim_c_parts: Dict[int, Dict[str, Any]] = {}
            sim_c_poses: Dict[int, ContinuousSE3Pose] = {}
            sim_c_obbs: Dict[int, Tuple[OBB3D, ...]] = {}
            cur_grav_c = (0.0, 0.0, -1.0)
            dirs_6_c = (
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
            )
            for op_c in artifact.execution_operations:
                op_tc = str(op_c.get("op_type", ""))
                if op_tc == "PLACE_PART":
                    iids_c = op_c.get("instance_ids", [])
                    if len(iids_c) != 1:
                        prefix_ok = False
                        reasons.append("malformed_trace_op:SE3_PLACE_PART")
                        break
                    iid_s = int(iids_c[0])
                    if iid_s in sim_c_parts:
                        prefix_ok = False
                        reasons.append(f"duplicate_trace_instance:{iid_s}")
                        break
                    raw_pid_c = str(op_c.get("part_id", "")).strip()
                    if raw_pid_c:
                        if not is_known_part_id(raw_pid_c):
                            prefix_ok = False
                            reasons.append(f"unknown_trace_part_id:{raw_pid_c}")
                            break
                        rmat_op = op_c.get("rot_matrix")
                        if rmat_op is None and iid_s in parts_by_id:
                            rmat_op = parts_by_id[iid_s]["rot_matrix"]
                        if rmat_op is None:
                            rmat_op = rot_id_to_matrix(int(op_c.get("rot_id", 0))).tolist()
                        pose_s = ContinuousSE3Pose.from_matrix_and_pos(
                            np.asarray(rmat_op, dtype=np.float64),
                            tuple(float(x) for x in op_c["pos"]),
                        )
                        pe_s = {
                            "instance_id": iid_s,
                            "part_id": raw_pid_c,
                            "rot_matrix": [[float(v) for v in row] for row in pose_s.rot_matrix],
                            "pos": [float(v) for v in pose_s.pos],
                            "is_grounded": bool(op_c.get("is_grounded", False)),
                            "color_id": int(
                                op_c.get(
                                    "color_id",
                                    parts_by_id[iid_s].get("color_id", 15) if iid_s in parts_by_id else 15,
                                )
                            ),
                        }
                    elif iid_s in parts_by_id:
                        pe_s = dict(parts_by_id[iid_s])
                        pose_s = poses_all[iid_s]
                    else:
                        prefix_ok = False
                        reasons.append(f"missing_trace_part_spec:{iid_s}")
                        break
                    pr_s = get_brick_primitive(str(pe_s["part_id"]))
                    obbs_s = tuple(box_to_obb(b, pose_s) for b in pr_s.hull_boxes)
                    ex_obbs_c = [eo for o_tup in sim_c_obbs.values() for eo in o_tup]
                    if any(obb_intersects_sat(so, eo, allowed_penetration_ldu=0.25) for eo in ex_obbs_c for so in obbs_s):
                        csg_ok = False
                        prefix_ok = False
                        reasons.append(f"trace_step_collision:{iid_s}")
                        break
                    if obl.require_monotone_insertion and ex_obbs_c:
                        corr_c = any(
                            not any(_swept_obb_hits_half_ray(so, eo, dv, 0.25) for eo in ex_obbs_c for so in obbs_s)
                            for dv in dirs_6_c
                        )
                        if not corr_c:
                            ins_ok = False
                    sim_c_parts[iid_s] = pe_s
                    sim_c_poses[iid_s] = pose_s
                    sim_c_obbs[iid_s] = obbs_s
                elif op_tc == "PLACE_SUBASSEMBLY":
                    sub_iids_c = [int(x) for x in op_c.get("instance_ids", [])]
                    if not sub_iids_c or any(sid in sim_c_parts or sid not in parts_by_id for sid in sub_iids_c):
                        prefix_ok = False
                        reasons.append("malformed_subassembly_instance_ids")
                        break
                    ex_obbs_c = [eo for o_tup in sim_c_obbs.values() for eo in o_tup]
                    grp_obbs_c = [so for sid in sub_iids_c for so in obbs_all[sid]]
                    if any(obb_intersects_sat(so, eo, allowed_penetration_ldu=0.25) for eo in ex_obbs_c for so in grp_obbs_c):
                        csg_ok = False
                        prefix_ok = False
                        reasons.append("trace_subassembly_collision")
                        break
                    if obl.require_monotone_insertion and ex_obbs_c:
                        corr_g = any(
                            not any(_swept_obb_hits_half_ray(so, eo, dv, 0.25) for eo in ex_obbs_c for so in grp_obbs_c)
                            for dv in dirs_6_c
                        )
                        if not corr_g:
                            ins_ok = False
                    for sid in sub_iids_c:
                        sim_c_parts[sid] = dict(parts_by_id[sid])
                        sim_c_poses[sid] = poses_all[sid]
                        sim_c_obbs[sid] = obbs_all[sid]
                elif op_tc == "REORIENT_ASSEMBLY":
                    gu_c = op_c.get("gravity_unit", [0.0, 0.0, -1.0])
                    if len(gu_c) != 3 or math.sqrt(sum(float(x) * float(x) for x in gu_c)) < 1e-6:
                        prefix_ok = False
                        reasons.append("invalid_reorient_gravity")
                        break
                    cur_grav_c = (float(gu_c[0]), float(gu_c[1]), float(gu_c[2]))
                else:
                    prefix_ok = False
                    reasons.append(f"unknown_op_type:{op_tc}")
                    break

                if obl.require_prefix_mechanics and sim_c_parts:
                    _check_deadline()
                    st_c_step, _, eq_rc, ub_vc, _ = _independent_continuous_se3_equilibrium_lp(
                        sim_c_parts,
                        sim_c_poses,
                        sim_c_obbs,
                        eff_foundation_policy,
                        eff_joint_assumptions,
                        gravity_unit=cur_grav_c,
                        ground_plane_z_ldu=eff_ground_z,
                    )
                    max_eq_r = max(max_eq_r, eq_rc)
                    max_ub_v = max(max_ub_v, ub_vc)
                    if not st_c_step:
                        prefix_ok = False
                        break

            if prefix_ok and ins_ok:
                if set(sim_c_parts.keys()) != set(parts_by_id.keys()):
                    prefix_ok = False
                    reasons.append("trace_terminal_instance_mismatch")
                else:
                    for iid_chk, w_e in parts_by_id.items():
                        s_e = sim_c_parts[iid_chk]
                        if (
                            str(s_e["part_id"]) != str(w_e["part_id"])
                            or bool(s_e.get("is_grounded", False)) != bool(w_e.get("is_grounded", False))
                            or int(s_e.get("color_id", 15)) != int(w_e.get("color_id", 15))
                            or not np.allclose(
                                np.asarray(s_e["rot_matrix"], dtype=np.float64),
                                np.asarray(w_e["rot_matrix"], dtype=np.float64),
                                atol=1e-4,
                            )
                            or not np.allclose(
                                np.asarray(s_e["pos"], dtype=np.float64),
                                np.asarray(w_e["pos"], dtype=np.float64),
                                atol=1e-4,
                            )
                        ):
                            prefix_ok = False
                            reasons.append(f"trace_terminal_state_mismatch:{iid_chk}")
                            break

        if obl.require_static_mechanics:
            _check_deadline()
            static_ok, final_eta, eq_rf, ub_vf, disc_pairs = _independent_continuous_se3_equilibrium_lp(
                parts_by_id,
                poses_all,
                obbs_all,
                eff_foundation_policy,
                eff_joint_assumptions,
                ground_plane_z_ldu=eff_ground_z,
            )
            max_eq_r = max(max_eq_r, eq_rf)
            max_ub_v = max(max_ub_v, ub_vf)

            # Verify lateral perturbation & loop-closure claims when present or required (`F2`)
            if static_ok and (
                claims.get("perturbation_mechanics") == "PASS" or obl.require_perturbation_if_articulated
            ):
                for lx, ly in ((0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05)):
                    lat_ok, _, _, _, _ = _independent_continuous_se3_equilibrium_lp(
                        parts_by_id,
                        poses_all,
                        obbs_all,
                        eff_foundation_policy,
                        eff_joint_assumptions,
                        lateral_accel_xy=(lx, ly),
                        ground_plane_z_ldu=eff_ground_z,
                    )
                    if not lat_ok:
                        static_ok = False
                        reasons.append("se3_lateral_perturbation_failed")
                        break

            if static_ok and (claims.get("loop_closure") == "PASS" or obl.require_loop_closure_if_cyclic):
                for (ia_l, pa_l), (ib_l, pb_l) in artifact.matched_port_pairs:
                    sa_l = get_brick_primitive(str(parts_by_id[ia_l]["part_id"])).ports[pa_l]
                    sb_l = get_brick_primitive(str(parts_by_id[ib_l]["part_id"])).ports[pb_l]
                    wfa_l = poses_all[ia_l].compose(ContinuousSE3Pose.from_ldu_frame(sa_l.local_frame))
                    wfb_l = poses_all[ib_l].compose(ContinuousSE3Pose.from_ldu_frame(sb_l.local_frame))
                    lo_a, hi_a = get_connector_axial_slide_bounds_ldu(sa_l.connector_type)
                    lo_b, hi_b = get_connector_axial_slide_bounds_ldu(sb_l.connector_type)
                    max_pos_tol = max(1.5, min(hi_a, hi_b) + 1.0)
                    if float(np.linalg.norm(wfa_l.t - wfb_l.t)) > max_pos_tol:
                        ports_ok = False
                        reasons.append("loop_closure_position_residual_exceeded")
                        break

    if not ins_ok:
        reasons.append("monotone_insertion_corridor_blocked")
    if not prefix_ok:
        reasons.append("prefix_equilibrium_replay_failed")
    if not static_ok:
        reasons.append("static_equilibrium_replay_failed")

    claims_ok = not any(r.startswith("claimed_certificate_not_pass:") for r in reasons)
    target_ok = (target_ver is not False)
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
        and claims_ok
        and target_ok
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
        verification_scope=ver_scope,
        target_verified=target_ver,
        target_digest_verified=target_digest_ver,
        target_overlap_verified=target_overlap_ver,
        measured_target_iou=measured_iou,
        inventory_status=inv_status,
        mechanics_profile_sha256=mech_profile_sha,
    )


def verify_certificate_json_file(
    path: str | Path,
    obligations: Optional[VerificationObligations] = None,
) -> IndependentVerificationReport:
    """
    Safely size-bound, parse, and verify a certificate JSON file (`S5`),
    returning a structured failure report on oversized/non-regular files, malformed JSON,
    or runtime replay exceptions.
    """
    obl = obligations if obligations is not None else VerificationObligations()
    limits = obl.resource_limits if obl.resource_limits is not None else VerifierResourceLimits()
    try:
        fpath = Path(path)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(fpath), flags)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"Certificate path is not a regular file: {fpath}")
            if st.st_size > limits.max_json_bytes:
                raise VerifierBudgetExceededError(
                    f"json_bytes={st.st_size}>max_json_bytes={limits.max_json_bytes}"
                )
            chunks = []
            read_so_far = 0
            while True:
                blk = os.read(fd, 1024 * 1024)
                if not blk:
                    break
                read_so_far += len(blk)
                if read_so_far > limits.max_json_bytes:
                    raise VerifierBudgetExceededError(
                        f"json_bytes>{limits.max_json_bytes}"
                    )
                chunks.append(blk)
            raw_text = b"".join(chunks).decode("utf-8")
        finally:
            os.close(fd)
        raw_obj = json.loads(raw_text)
        if not isinstance(raw_obj, dict):
            raise ValueError(f"Expected top-level JSON object, got {type(raw_obj).__name__}")
        artifact = VersionedCertificateArtifact.from_json_dict(raw_obj)
        return verify_certificate_independently(artifact, obligations=obl)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return IndependentVerificationReport(
            verified=False,
            schema_version_ok=False,
            catalog_sha256_valid=False,
            inventory_valid=False,
            csg_2ldu_collision_free=False,
            ports_compatible=False,
            monotone_insertion_valid=False,
            static_mechanics_valid=False,
            prefix_mechanics_valid=False,
            witness_residuals_valid=False,
            max_eq_residual_g=float("inf"),
            max_ub_violation=float("inf"),
            replayed_peak_utilization=float("inf"),
            failure_reasons=(f"invalid_artifact_json:{type(exc).__name__}:{exc}",),
            verification_scope="invalid_artifact_json",
            target_verified=False,
            inventory_status="failed",
        )


def _isolated_worker_entry(
    json_path_str: str,
    obligations: Optional[VerificationObligations],
    memory_limit_mb: Optional[int],
    out_q: Any,
) -> None:
    if memory_limit_mb is not None and memory_limit_mb > 0:
        try:
            import resource

            limit_bytes = int(memory_limit_mb) * 1024 * 1024
            # RLIMIT_AS is enforced on Linux; ignore ValueError on macOS where RLIMIT_AS is unsupported
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            pass
    rep = verify_certificate_json_file(json_path_str, obligations=obligations)
    out_q.put(rep)


def verify_certificate_in_isolated_worker(
    path: str | Path,
    obligations: Optional[VerificationObligations] = None,
    *,
    timeout_sec: float = 60.0,
    memory_limit_mb: Optional[int] = 2048,
) -> IndependentVerificationReport:
    """
    Execute `verify_certificate_json_file` inside a fresh `spawn` worker process (`S5`)
    with hard wall-clock supervision and optional POSIX virtual-memory caps so an adversarial
    input or native solver fault cannot wedge or OOM a shared parent service.
    """
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_isolated_worker_entry,
        args=(str(Path(path)), obligations, memory_limit_mb, out_q),
        daemon=True,
    )
    proc.start()
    try:
        proc.join(timeout=max(0.05, float(timeout_sec)))
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)
            return IndependentVerificationReport(
                verified=False,
                schema_version_ok=False,
                catalog_sha256_valid=False,
                inventory_valid=False,
                csg_2ldu_collision_free=False,
                ports_compatible=False,
                monotone_insertion_valid=False,
                static_mechanics_valid=False,
                prefix_mechanics_valid=False,
                witness_residuals_valid=False,
                max_eq_residual_g=float("inf"),
                max_ub_violation=float("inf"),
                replayed_peak_utilization=float("inf"),
                failure_reasons=(f"worker_timeout_exceeded:{timeout_sec}s",),
                verification_scope="isolated_worker_failed",
                target_verified=False,
                inventory_status="failed",
            )
        if not out_q.empty():
            return out_q.get_nowait()
        return IndependentVerificationReport(
            verified=False,
            schema_version_ok=False,
            catalog_sha256_valid=False,
            inventory_valid=False,
            csg_2ldu_collision_free=False,
            ports_compatible=False,
            monotone_insertion_valid=False,
            static_mechanics_valid=False,
            prefix_mechanics_valid=False,
            witness_residuals_valid=False,
            max_eq_residual_g=float("inf"),
            max_ub_violation=float("inf"),
            replayed_peak_utilization=float("inf"),
            failure_reasons=(f"worker_exited_without_report:exitcode={proc.exitcode}",),
            verification_scope="isolated_worker_failed",
            target_verified=False,
            inventory_status="failed",
        )
    finally:
        out_q.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Planner-free independent certificate verifier (`topo_brick`, R9-WS1 / Phase 13). "
            "Without --expected-target-sha256, runs assembly_only verification; pass relying-party "
            "flags to enforce target digest, license, foundation, or inventory policies."
        )
    )
    parser.add_argument("artifact_json", nargs="?", help="Path to VersionedCertificateArtifact JSON")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run standalone 2-LDU CSG collider audit + synthetic certificate replay check",
    )
    parser.add_argument("--expected-target-sha256", default=None, help="Relying-party expected target SHA-256")
    parser.add_argument(
        "--expected-license-filter",
        choices=("all", "cc_by_4_0_only"),
        default=None,
        help="Relying-party license policy ('cc_by_4_0_only' or 'all')",
    )
    parser.add_argument(
        "--expected-foundation-policy",
        choices=_VALID_FOUNDATION_POLICIES,
        default=None,
        help="Relying-party expected foundation policy",
    )
    parser.add_argument(
        "--expected-ground-plane-z-ldu",
        default="0.0",
        help="Relying-party expected ground plane Z in LDU (default: 0.0; pass 'any' to accept certificate datum)",
    )
    parser.add_argument(
        "--expected-inventory-json",
        type=Path,
        default=None,
        help="Path to JSON object mapping mold_id -> max allowed count",
    )
    parser.add_argument(
        "--require-explicit-inventory",
        action="store_true",
        help="Fail closed if neither relying-party nor certificate inventory limits are present",
    )
    parser.add_argument(
        "--max-json-bytes",
        type=int,
        default=32 * 1024 * 1024,
        help="Maximum allowed certificate JSON file size in bytes (default: 32 MiB)",
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

    exp_inv: Optional[Dict[str, int]] = None
    if args.expected_inventory_json is not None:
        raw_inv = json.loads(args.expected_inventory_json.read_text(encoding="utf-8"))
        if not isinstance(raw_inv, dict):
            parser.error("--expected-inventory-json must contain a JSON object.")
        exp_inv = {str(k): int(v) for k, v in raw_inv.items()}

    raw_gz_arg = str(args.expected_ground_plane_z_ldu).strip().lower()
    if raw_gz_arg == "any":
        exp_gz_cli: Optional[float] = None
    else:
        try:
            exp_gz_cli = float(raw_gz_arg)
        except ValueError:
            parser.error(f"Invalid --expected-ground-plane-z-ldu: {args.expected_ground_plane_z_ldu!r}")

    cli_obl = VerificationObligations(
        expected_target_sha256=args.expected_target_sha256,
        expected_license_filter=args.expected_license_filter,
        expected_foundation_policy=args.expected_foundation_policy,
        expected_inventory_counts=exp_inv,
        require_explicit_inventory_spec=bool(args.require_explicit_inventory),
        resource_limits=VerifierResourceLimits(max_json_bytes=int(args.max_json_bytes)),
        expected_ground_plane_z_ldu=exp_gz_cli,
    )
    rep = verify_certificate_json_file(args.artifact_json, obligations=cli_obl)
    print(json.dumps(asdict(rep), indent=2))
    return 0 if rep.verified else 1


if __name__ == "__main__":
    sys.exit(main())
