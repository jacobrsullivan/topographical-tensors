"""
Same-target cross-route comparison harness (`Phase 15 Priority 2`):
Exercises one geometric target across `course_by_course_search`, `open_catalog_sculpted`,
and `compositional_mpd`, evaluating actual exported `.mpd` files across 5 canonical camera
views alongside `RealizeRouteProvenance` and `"assembly_and_target"` independent replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from topo_brick.evaluation.independent_verifier import (
    VersionedCertificateArtifact,
    build_relying_party_obligations,
    export_certificate_artifact,
    verify_certificate_independently,
)
from topo_brick.interface.ldraw_shadow_catalog_compiler import COMBINED_BRICK_CATALOG
from topo_brick.interface.ldu_frames import LDUBox, rot_id_to_matrix
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.interface.se3_kinematics import (
    ContinuousSE3Pose,
    flatten_omr_mpd_with_diagnostics,
)
from topo_brick.interface.tier_a_catalog import normalize_ldraw_part_id
from topo_brick.kernel.adaptive_surface_and_macro_planner import (
    AdaptiveSurfaceTarget,
    evaluate_exact_5x5x2_mesh_iou,
)
from topo_brick.kernel.construction_kernel import TopoBrickKernel
from topo_brick.kernel.kernel_contracts import RealizeRouteProvenance
from topo_brick.kernel.subassembly_realizer import SubmodelNodeSpec
from topo_brick.kernel.view_priority import ViewPrioritySpec, evaluate_camera_reprojection_fidelity
from topo_brick.representation.interface_fields import SparseTiledTargetField, TargetShapeBoxes

CANONICAL_FIVE_CAMERA_VIEWS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("front_pos_y", (0.0, 1.0, 0.0)),
    ("side_pos_x", (1.0, 0.0, 0.0)),
    ("top_neg_z", (0.0, 0.0, -1.0)),
    ("oblique_ne", (1.0, 1.0, -0.55)),
    ("oblique_sw", (-1.0, -1.0, -0.55)),
)


@dataclass(frozen=True, slots=True)
class MPDRoundTripVerificationReport:
    """
    Fail-closed round-trip verification report over delivered `.mpd` text (`Phase 16 Decision 3`),
    auditing silent drops, unknown part references, collision rejections, and bijective
    `(canonical_part_id, ContinuousSE3Pose, color_id)` parity against `AssemblyState` / certificates.
    """

    total_mpd_instances: int
    reconstructed_cubic_instances: int
    non_cubic_se3_instances: int
    unknown_part_refs: Tuple[str, ...]
    collision_failures: int
    dropped_instances_count: int
    exact_part_pose_color_parity: bool
    roundtrip_verified: bool
    failure_reasons: Tuple[str, ...] = ()
    non_rigid_transform_failures: int = 0
    unresolved_color_failures: int = 0


@dataclass(frozen=True, slots=True)
class SingleRouteEvaluationEntry:
    """Unified evaluation summary for one route executed on a shared benchmark target."""

    route_label: str
    status: str
    route_provenance: RealizeRouteProvenance
    num_parts: int
    distinct_molds: int
    wall_clock_ms: float
    exported_mpd_source: str
    exported_mpd_part_count: int
    caller_target_iou: float
    evaluated_target_iou: float
    mesh_5x5x2_iou: float
    exported_five_view_silhouette_iou: Dict[str, float]
    mean_five_view_silhouette_iou: float
    independent_verified: bool
    verification_scope: str
    mpd_roundtrip_report: Optional[MPDRoundTripVerificationReport] = None
    mpd_roundtrip_verified: bool = True
    exported_five_view_color_delta_e00: Dict[str, Optional[float]] = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class CrossRouteComparisonReport:
    """Side-by-side comparison of course, sculpted, and compositional routes on one target."""

    caller_target_sha256: str
    entries_by_route: Dict[str, SingleRouteEvaluationEntry]


def _so3_angle_error_rad(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r_rel = r_a.T @ r_b
    skew = math.sqrt(
        max(
            0.0,
            float(
                (r_rel[2, 1] - r_rel[1, 2]) ** 2
                + (r_rel[0, 2] - r_rel[2, 0]) ** 2
                + (r_rel[1, 0] - r_rel[0, 1]) ** 2
            ),
        )
    )
    return float(math.atan2(0.5 * skew, 0.5 * (float(np.trace(r_rel)) - 1.0)))


def _count_malformed_type1_lines(mpd_text: str) -> int:
    malformed = 0
    for ln in mpd_text.splitlines():
        s = ln.strip()
        if not s.startswith("1 "):
            continue
        toks = s.split()
        if len(toks) < 15:
            malformed += 1
            continue
        try:
            int(toks[1])
            for x in toks[2:14]:
                val = float(x)
                if not math.isfinite(val):
                    malformed += 1
                    break
        except ValueError:
            malformed += 1
    return malformed


def _bijective_se3_color_match(
    actual_items: Sequence[Tuple[str, ContinuousSE3Pose, int]],
    expected_items: Sequence[Tuple[str, ContinuousSE3Pose, int]],
    *,
    pos_tol_ldu: float = 0.05,
    rot_tol_rad: float = 1.0e-3,
    label: str = "assembly",
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if len(actual_items) != len(expected_items):
        reasons.append(
            f"{label}_count_mismatch: mpd={len(actual_items)} vs expected={len(expected_items)}"
        )
    used_exp: set[int] = set()
    for idx_a, (cid_a, pose_a, col_a) in enumerate(actual_items):
        best_j: Optional[int] = None
        best_score: Tuple[float, float] = (float("inf"), float("inf"))
        t_a = pose_a.t
        r_a = pose_a.R
        for idx_e, (cid_e, pose_e, col_e) in enumerate(expected_items):
            if idx_e in used_exp or cid_a != cid_e or int(col_a) != int(col_e):
                continue
            p_err = float(np.linalg.norm(t_a - pose_e.t))
            if p_err > pos_tol_ldu:
                continue
            r_err = _so3_angle_error_rad(r_a, pose_e.R)
            if r_err > rot_tol_rad:
                continue
            sc = (p_err, r_err)
            if sc < best_score:
                best_score = sc
                best_j = idx_e
        if best_j is not None:
            used_exp.add(best_j)
        else:
            reasons.append(
                f"{label}_unmatched_mpd_instance[{idx_a}]: part={cid_a}, color={col_a}, pos={tuple(round(v, 3) for v in pose_a.pos)}"
            )
    for idx_e, (cid_e, pose_e, col_e) in enumerate(expected_items):
        if idx_e not in used_exp:
            reasons.append(
                f"{label}_missing_expected_part[{idx_e}]: part={cid_e}, color={col_e}, pos={tuple(round(v, 3) for v in pose_e.pos)}"
            )
    return (len(reasons) == 0), reasons


def verify_exported_mpd_roundtrip(
    mpd_text: str,
    *,
    expected_assembly: Optional[Any] = None,
    certificate_artifact: Optional[VersionedCertificateArtifact] = None,
    ground_plane_z_ldu: float = 0.0,
    pos_tol_ldu: float = 0.05,
    rot_tol_rad: float = 1.0e-3,
) -> Tuple[AssemblyState, MPDRoundTripVerificationReport]:
    """
    Parse a delivered `.mpd` / `.ldr` text, rebuild its cubic `AssemblyState` without silent drops,
    and verify bijective `(canonical_part_id, ContinuousSE3Pose, color_id)` parity against
    `expected_assembly` and/or `certificate_artifact.placed_parts` (`Phase 16 Decision 3`).
    """
    failure_reasons: List[str] = []
    malformed_cnt = _count_malformed_type1_lines(mpd_text)
    if malformed_cnt > 0:
        failure_reasons.append(f"malformed_type1_lines: {malformed_cnt}")

    flattened, struct_errs = flatten_omr_mpd_with_diagnostics(
        mpd_text,
        catalog=COMBINED_BRICK_CATALOG,
        strict_rigid_transforms=True,
        so3_tol=max(2.0e-3, float(rot_tol_rad) * 2.0),
    )
    if struct_errs:
        failure_reasons.extend(struct_errs)
    total_mpd = len(flattened) + malformed_cnt
    state = AssemblyState()

    unknown_refs: List[str] = []
    non_cubic_cnt = 0
    non_rigid_cnt = sum(1 for e in struct_errs if e.startswith("non_rigid_submodel_transform:"))
    unresolved_col_cnt = 0
    collision_cnt = 0
    reconstructed_cnt = 0
    dropped_cnt = malformed_cnt + len(struct_errs)
    parsed_se3_items: List[Tuple[str, ContinuousSE3Pose, int]] = []

    expect_continuous_only = (
        expected_assembly is not None and not isinstance(expected_assembly, AssemblyState)
    ) or (
        certificate_artifact is not None
        and str(certificate_artifact.assembly_mode) == "continuous_se3"
    )

    for idx, pl in enumerate(flattened):
        if pl.canonical_part_id not in COMBINED_BRICK_CATALOG:
            unknown_refs.append(str(pl.raw_part_ref))
            dropped_cnt += 1
            failure_reasons.append(
                f"unknown_or_unsupported_part[{idx}]: raw={pl.raw_part_ref!r}, canonical={pl.canonical_part_id!r}"
            )
            continue
        if not pl.is_rigid_so3 or pl.continuous_pose is None:
            non_rigid_cnt += 1
            dropped_cnt += 1
            r_arr = np.asarray(pl.ldraw_world_R, dtype=np.float64)
            det_v = float(np.linalg.det(r_arr)) if r_arr.shape == (3, 3) else float("nan")
            failure_reasons.append(
                f"non_rigid_transform[{idx}]: raw={pl.raw_part_ref!r}, det={det_v:.6g}"
            )
            continue
        if int(pl.color_code) in (16, 24) or int(pl.color_code) < 0:
            unresolved_col_cnt += 1
            dropped_cnt += 1
            failure_reasons.append(
                f"unresolved_meta_color[{idx}]: part={pl.canonical_part_id}, color={pl.color_code}"
            )
            continue

        parsed_se3_items.append(
            (str(pl.canonical_part_id), pl.continuous_pose, int(pl.color_code))
        )
        fr = pl.continuous_pose.to_ldu_frame_if_cubic(tol=rot_tol_rad)
        if fr is None:
            non_cubic_cnt += 1
            if not expect_continuous_only:
                dropped_cnt += 1
                failure_reasons.append(
                    f"non_cubic_pose_in_cubic_export[{idx}]: part={pl.canonical_part_id}"
                )
            continue

        wbs = [b.transform(fr) for b in COMBINED_BRICK_CATALOG[pl.canonical_part_id].hull_boxes]
        min_z = float(min(b.zmin for b in wbs)) if wbs else float(fr.pos[2])
        try:
            state.place_root_part(
                pl.canonical_part_id,
                fr,
                is_grounded=(abs(min_z - float(ground_plane_z_ldu)) <= 0.25),
                color_id=int(pl.color_code),
            )
            reconstructed_cnt += 1
        except ValueError as exc:
            collision_cnt += 1
            dropped_cnt += 1
            failure_reasons.append(
                f"collision_on_mpd_replay[{idx}]: part={pl.canonical_part_id} at {fr.pos}: {exc}"
            )

    state.mark_untraced()

    parity_ok = len(failure_reasons) == 0
    if expected_assembly is not None:
        exp_asm_items: List[Tuple[str, ContinuousSE3Pose, int]] = []
        for iid in sorted(expected_assembly.placed_parts.keys()):
            p = expected_assembly.placed_parts[iid]
            cid = normalize_ldraw_part_id(str(p.part_id), COMBINED_BRICK_CATALOG)
            if hasattr(p, "continuous_pose") and getattr(p, "continuous_pose") is not None:
                c_pose = p.continuous_pose
            else:
                c_pose = ContinuousSE3Pose.from_ldu_frame(p.pose)
            exp_asm_items.append((cid, c_pose, int(getattr(p, "color_id", 15))))
        asm_ok, asm_reasons = _bijective_se3_color_match(
            parsed_se3_items,
            exp_asm_items,
            pos_tol_ldu=pos_tol_ldu,
            rot_tol_rad=rot_tol_rad,
            label="assembly",
        )
        if not asm_ok:
            parity_ok = False
            failure_reasons.extend(asm_reasons)

    if certificate_artifact is not None:
        exp_cert_items: List[Tuple[str, ContinuousSE3Pose, int]] = []
        for entry in certificate_artifact.placed_parts:
            cid = normalize_ldraw_part_id(str(entry["part_id"]), COMBINED_BRICK_CATALOG)
            pos_t = tuple(float(v) for v in entry["pos"])
            if "rot_matrix" in entry and entry["rot_matrix"] is not None:
                r_mat = np.asarray(entry["rot_matrix"], dtype=np.float64)
            else:
                r_mat = np.asarray(rot_id_to_matrix(int(entry["rot_id"])), dtype=np.float64)
            c_pose = ContinuousSE3Pose.from_matrix_and_pos(r_mat, pos_t, orthonormalize=False)
            exp_cert_items.append((cid, c_pose, int(entry.get("color_id", 15))))
        cert_ok, cert_reasons = _bijective_se3_color_match(
            parsed_se3_items,
            exp_cert_items,
            pos_tol_ldu=pos_tol_ldu,
            rot_tol_rad=rot_tol_rad,
            label="certificate",
        )
        if not cert_ok:
            parity_ok = False
            failure_reasons.extend(cert_reasons)

    verified = (
        len(unknown_refs) == 0
        and non_rigid_cnt == 0
        and unresolved_col_cnt == 0
        and collision_cnt == 0
        and dropped_cnt == 0
        and parity_ok
        and len(failure_reasons) == 0
    )
    report = MPDRoundTripVerificationReport(
        total_mpd_instances=total_mpd,
        reconstructed_cubic_instances=reconstructed_cnt,
        non_cubic_se3_instances=non_cubic_cnt,
        unknown_part_refs=tuple(unknown_refs),
        collision_failures=collision_cnt,
        dropped_instances_count=dropped_cnt,
        exact_part_pose_color_parity=bool(parity_ok),
        roundtrip_verified=bool(verified),
        failure_reasons=tuple(failure_reasons),
        non_rigid_transform_failures=non_rigid_cnt,
        unresolved_color_failures=unresolved_col_cnt,
    )
    return state, report


def _rebuild_cubic_state_from_exported_mpd(
    mpd_text: str,
    ground_plane_z_ldu: float = 0.0,
) -> AssemblyState:
    state, _ = verify_exported_mpd_roundtrip(
        mpd_text,
        ground_plane_z_ldu=ground_plane_z_ldu,
    )
    return state


def compare_realize_routes_on_target(
    target_field: SparseTiledTargetField | TargetShapeBoxes,
    adaptive_target: AdaptiveSurfaceTarget,
    compositional_nodes: Sequence[SubmodelNodeSpec],
    *,
    kernel: Optional[TopoBrickKernel] = None,
    license_filter: str = "cc_by_4_0_only",
    forbidden_support_boxes: Tuple[LDUBox, ...] = (),
    verts_ldu: Optional[np.ndarray] = None,
    vertex_colors_rgb: Optional[np.ndarray] = None,
    foundation_policy: Any = "clutched_baseplate",
    ground_plane_z_ldu: float = 0.0,
) -> CrossRouteComparisonReport:
    """
    Exercise one target across `course_by_course_search`, `open_catalog_sculpted`, and
    `compositional_mpd`, comparing actual exported `.mpd` views, recall thresholds, and
    `"assembly_and_target"` independent replay.
    """
    kern = kernel or TopoBrickKernel()

    runs = (
        (
            "course_by_course_search",
            kern.realize(
                target_field,
                solver_id="S1_CoverAndRepair",
                support_policy="minimal_permanent_pillars",
                forbidden_support_boxes=forbidden_support_boxes,
                license_filter=license_filter,
                foundation_policy=foundation_policy,
                ground_plane_z_ldu=ground_plane_z_ldu,
            ),
        ),
        (
            "open_catalog_sculpted",
            kern.realize(
                target_field,
                part_retrieval="open_catalog_top_k",
                surface_refinement="open_catalog_sculpted",
                adaptive_mesh_target=adaptive_target,
                support_policy="minimal_permanent_pillars",
                forbidden_support_boxes=forbidden_support_boxes,
                license_filter=license_filter,
                verts_ldu=verts_ldu,
                vertex_colors_rgb=vertex_colors_rgb,
                foundation_policy=foundation_policy,
                ground_plane_z_ldu=ground_plane_z_ldu,
            ),
        ),
        (
            "compositional_mpd",
            kern.realize(
                target_field,
                subassembly_graph=compositional_nodes,
                support_policy="minimal_permanent_pillars",
                forbidden_support_boxes=forbidden_support_boxes,
                license_filter=license_filter,
                verts_ldu=verts_ldu,
                vertex_colors_rgb=vertex_colors_rgb,
                foundation_policy=foundation_policy,
                ground_plane_z_ldu=ground_plane_z_ldu,
            ),
        ),
    )

    entries: Dict[str, SingleRouteEvaluationEntry] = {}
    caller_sha = ""
    for label, res in runs:
        rp = res.route_provenance
        if rp is None:
            raise RuntimeError(f"Missing route_provenance on route {label!r}")
        caller_sha = rp.caller_target_sha256

        if res.compositional_mpd_text is not None:
            mpd_text = res.compositional_mpd_text
            mpd_src = "compositional_mpd_text"
        else:
            mpd_text = res.assembly.to_ldraw_mpd(model_name=f"{label}.ldr")
            mpd_src = "assembly.to_ldraw_mpd"

        eval_tf = res.evaluated_target_field if res.evaluated_target_field is not None else target_field
        art = export_certificate_artifact(
            res.assembly,
            res.certificates,
            validator=kern.validator,
            target_field=eval_tf,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=forbidden_support_boxes,
            license_filter=license_filter,
            foundation_policy=rp.foundation_policy,
            ground_plane_z_ldu=rp.ground_plane_z_ldu,
        )
        exported_state, mpd_rt_rep = verify_exported_mpd_roundtrip(
            mpd_text,
            expected_assembly=res.assembly,
            certificate_artifact=art,
            ground_plane_z_ldu=rp.ground_plane_z_ldu,
        )
        mesh_iou = float(evaluate_exact_5x5x2_mesh_iou(adaptive_target, exported_state))

        ad_origin = getattr(adaptive_target, "origin_ldu", (0, 0, 0))
        view_scores: Dict[str, float] = {}
        view_color_de00: Dict[str, Optional[float]] = {}
        for v_name, v_dir in CANONICAL_FIVE_CAMERA_VIEWS:
            v_rep = evaluate_camera_reprojection_fidelity(
                state=exported_state,
                occ_5x5x6=adaptive_target.occ_5x5x6,
                spec=ViewPrioritySpec(view_dir_ldu=v_dir),
                verts_ldu=verts_ldu,
                vertex_colors_rgb=vertex_colors_rgb,
                origin_ldu=ad_origin,
            )
            view_scores[v_name] = float(v_rep.silhouette_iou_2d)
            view_color_de00[v_name] = (
                float(v_rep.visible_pixel_mean_delta_e00)
                if (verts_ldu is not None and vertex_colors_rgb is not None and len(vertex_colors_rgb) > 0)
                else None
            )

        obl = build_relying_party_obligations(
            expected_target_field=eval_tf,
            min_target_iou=max(0.10, rp.min_recall_threshold * 0.5),
            forbidden_support_boxes=forbidden_support_boxes,
            license_filter=license_filter,
            foundation_policy=rp.foundation_policy,
            ground_plane_z_ldu=rp.ground_plane_z_ldu,
        )
        ver = verify_certificate_independently(art, obligations=obl)

        entries[label] = SingleRouteEvaluationEntry(
            route_label=label,
            status=res.status.value,
            route_provenance=rp,
            num_parts=int(res.assembly.num_parts),
            distinct_molds=len({p.part_id for p in res.assembly.placed_parts.values()}),
            wall_clock_ms=float(res.build_record.wall_clock_ms),
            exported_mpd_source=mpd_src,
            exported_mpd_part_count=int(exported_state.num_parts),
            caller_target_iou=float(rp.achieved_caller_target_iou),
            evaluated_target_iou=float(rp.achieved_evaluated_iou),
            mesh_5x5x2_iou=mesh_iou,
            exported_five_view_silhouette_iou=view_scores,
            mean_five_view_silhouette_iou=float(np.mean(list(view_scores.values()))),
            independent_verified=bool(ver.verified and mpd_rt_rep.roundtrip_verified),
            verification_scope=str(ver.verification_scope),
            mpd_roundtrip_report=mpd_rt_rep,
            mpd_roundtrip_verified=bool(mpd_rt_rep.roundtrip_verified),
            exported_five_view_color_delta_e00=view_color_de00,
        )

    return CrossRouteComparisonReport(caller_target_sha256=caller_sha, entries_by_route=entries)
