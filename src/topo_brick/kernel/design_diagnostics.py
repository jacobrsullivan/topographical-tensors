"""
Non-certificate planner diagnostics for catalog funnel coverage, regional/view-weighted
appearance (`CIEDE2000`), colour-boundary straddling, and bilateral construction symmetry
(`Phase 15–16`).

Design scores here are strictly diagnostic telemetry and NEVER enter `CertificateBundle`
or `independent_verifier.py` predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.interface.brick_catalog import EXTENDED_BRICK_CATALOG
from topo_brick.interface.catalog_retrieval_index import (
    CATALOG_RETRIEVAL_INDEX,
    SCULPTURAL_TAXONOMY_CLASSES,
    CatalogRetrievalResult,
)
from topo_brick.interface.ldraw_palette import LDRAW_FULL_PALETTE_RECORDS
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState

# Verified opposite-hand LDraw wedge / wing mold pairs present in `COMBINED_BRICK_CATALOG`
KNOWN_HANDED_COUNTERPART_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("43722", "43723"),
    ("41769", "41770"),
    ("24299", "24307"),
    ("29119", "29120"),
    ("54383", "54384"),
    ("41747", "41748"),
    ("47397", "47398"),
    ("50304", "50305"),
    ("6564", "6565"),
    ("2419", "2450"),
    ("41767", "41768"),
    ("50955", "50956"),
)

HANDED_COUNTERPART_MAP: Dict[str, str] = {
    **{a: b for a, b in KNOWN_HANDED_COUNTERPART_PAIRS},
    **{b: a for a, b in KNOWN_HANDED_COUNTERPART_PAIRS},
}


@dataclass(frozen=True, slots=True)
class CatalogFunnelAndShortlistReport:
    """Explicit catalog eligibility, shortlist, and placed mold-family telemetry (`Priority 4`)."""

    combined_catalog_ids: int
    compiled_catalog_ids: int
    canonical_system_molds: int
    active_buildable_molds: int
    valid_sculptural_molds: int
    pose_signatures_total: int
    license_filter: str
    license_and_kit_eligible_molds: int
    shortlist_molds: Tuple[str, ...]
    shortlist_pose_signatures_scored: int
    backbone_molds_included: Tuple[str, ...]
    placed_mold_family_counts: Dict[str, int]
    shortlist_c4_footprints_available: int = 0
    actual_candidate_poses_scored: int = 0


@dataclass(frozen=True, slots=True)
class RegionalAppearanceReport:
    """Open/finite colour coverage, visible-vs-occluded error, regional `ΔE_00`, and boundary straddling."""

    palette_mode: str
    palette_candidates_searched: int
    visible_skin_mean_delta_e00: Optional[float]
    occluded_core_mean_delta_e00: Optional[float]
    per_region_mean_delta_e00: Dict[str, float]
    high_gradient_boundary_bricks_count: int
    high_gradient_instance_ids: Tuple[int, ...]
    color_samples_available: bool = False
    sampled_skin_parts_count: int = 0
    sampled_core_parts_count: int = 0


@dataclass(frozen=True, slots=True)
class BilateralSymmetryReport:
    """
    Bilateral construction-symmetry audit separating 2D/3D silhouette mirror overlap and internal
    core filler from true exterior skin mold / handed-wedge / seam correspondence (`Phase 16`).
    """

    symmetry_axis: str
    plane_coord_ldu: float
    total_parts: int
    centerline_spanning_count: int
    exact_same_mold_pairs: int
    handed_counterpart_pairs: int
    unmatched_parts_count: int
    exterior_construction_symmetry_ratio: Optional[float]
    seam_alignment_ratio: Optional[float]
    silhouette_mirror_iou: float
    symmetric_color_agreement_ratio: Optional[float]
    comparable_pairs_count: int = 0
    exterior_non_centerline_parts_count: int = 0
    exterior_paired_parts_count: int = 0
    exterior_unmatched_parts_count: int = 0
    overall_construction_symmetry_ratio: Optional[float] = None
    volumetric_mirror_iou_5x5x2: float = 0.0
    orthographic_silhouette_mirror_iou: float = 0.0
    per_region_symmetry_ratio: Dict[str, Optional[float]] = field(default_factory=dict)


def build_catalog_funnel_and_shortlist_report(
    *,
    state: AssemblyState,
    retrieval_result: CatalogRetrievalResult,
    license_filter: str = "all",
    allowed_part_ids: Optional[Sequence[str]] = None,
    plate_to_tile_values: Sequence[str] = (),
    actual_candidate_poses_scored: int = 0,
) -> CatalogFunnelAndShortlistReport:
    """Construct exact catalog funnel denominators + shortlist & mold-family counts for a run."""
    meta = CATALOG_RETRIEVAL_INDEX.funnel_metadata()
    allowed_set = set(str(p) for p in allowed_part_ids) if allowed_part_ids is not None else None

    eligible_cnt = 0
    shortlist_set = set(retrieval_result.retrieved_part_ids)
    shortlist_poses = 0
    for ent in CATALOG_RETRIEVAL_INDEX.buildable_entries:
        if license_filter == "cc_by_4_0_only" and ent.license_tier != "cc_by_4_0":
            continue
        if allowed_set is not None and ent.mold_id not in allowed_set:
            continue
        eligible_cnt += 1
        if ent.mold_id in shortlist_set:
            shortlist_poses += max(1, len(ent.c4_footprints))

    backbone_present = tuple(
        pid
        for pid in retrieval_result.essential_backbone_part_ids
        if (allowed_set is None or pid in allowed_set)
    )
    tile_set = set(plate_to_tile_values)
    family_counts: Dict[str, int] = {
        "brick_plate": 0,
        "slope_curved": 0,
        "wedge": 0,
        "snot": 0,
        "tile": 0,
        "jumper": 0,
        "landmark_other": 0,
    }
    for p in state.placed_parts.values():
        can_p = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(p.part_id) or p.part_id
        ent = CATALOG_RETRIEVAL_INDEX.get_entry(can_p)
        cls_p = ent.taxonomy_class if ent is not None else ""
        if p.part_id in tile_set or cls_p == "SMOOTH_TILE":
            family_counts["tile"] += 1
        elif cls_p == "JUMPER_HALF_STUD":
            family_counts["jumper"] += 1
        elif cls_p == "SNOT_ANCHOR" or LDUFrame(p.pose.rot_id, (0, 0, 0)).z_axis[2] == 0:
            family_counts["snot"] += 1
        elif "WEDGE" in cls_p or can_p in HANDED_COUNTERPART_MAP:
            family_counts["wedge"] += 1
        elif cls_p in SCULPTURAL_TAXONOMY_CLASSES:
            family_counts["slope_curved"] += 1
        elif p.part_id in EXTENDED_BRICK_CATALOG or cls_p in ("CORE_BRICK", "CORE_PLATE", ""):
            family_counts["brick_plate"] += 1
        else:
            family_counts["landmark_other"] += 1

    c4_avail = max(shortlist_poses, len(shortlist_set))
    act_scored = max(0, int(actual_candidate_poses_scored))
    return CatalogFunnelAndShortlistReport(
        combined_catalog_ids=int(meta["combined_catalog_ids"]),
        compiled_catalog_ids=int(meta["compiled_catalog_ids"]),
        canonical_system_molds=int(meta["canonical_system_molds"]),
        active_buildable_molds=int(meta["active_buildable_molds"]),
        valid_sculptural_molds=int(meta["valid_sculptural_molds"]),
        pose_signatures_total=int(meta["pose_signatures"]),
        license_filter=str(license_filter),
        license_and_kit_eligible_molds=eligible_cnt,
        shortlist_molds=tuple(retrieval_result.retrieved_part_ids),
        shortlist_pose_signatures_scored=(act_scored if act_scored > 0 else c4_avail),
        backbone_molds_included=backbone_present,
        placed_mold_family_counts=family_counts,
        shortlist_c4_footprints_available=c4_avail,
        actual_candidate_poses_scored=act_scored,
    )


def evaluate_regional_appearance_diagnostics(
    *,
    state: AssemblyState,
    verts_ldu: Optional[np.ndarray],
    vertex_colors_rgb: Optional[np.ndarray],
    vertex_region_labels: Optional[Sequence[Any]] = None,
    view_priority: Optional[Any] = None,
    colored_inventory: Optional[Any] = None,
    open_color_palette_mode: str = "solid_54_default",
    boundary_gradient_threshold_de00: float = 18.0,
) -> RegionalAppearanceReport:
    """
    Evaluate visible vs. occluded `CIEDE2000`, per-region `ΔE_00`, and boundary-straddling bricks
    (e.g., a single brick spanning a crisp red/white stripe).
    """
    from topo_brick.kernel.colored_inventory_allocator import (
        classify_part_surface_exposure,
    )
    from topo_brick.kernel.sculpted_primitives import (
        LDRAW_SOLID_PALETTE_LAB,
        LDRAW_SOLID_PALETTE_SRGB,
        ciede2000_delta_e,
        srgb_to_cielab,
    )

    if colored_inventory is not None:
        pal_mode = "finite_colored_inventory"
        pal_cnt = len({int(col) for (_, col), q in colored_inventory.counts.items() if int(q) > 0})
    elif open_color_palette_mode == "ldconfig_finish_aware":
        pal_mode = "ldconfig_finish_aware"
        pal_cnt = len(LDRAW_FULL_PALETTE_RECORDS)
    else:
        pal_mode = "solid_54_default"
        pal_cnt = len(LDRAW_SOLID_PALETTE_SRGB)

    if (
        state.num_parts == 0
        or verts_ldu is None
        or vertex_colors_rgb is None
        or len(verts_ldu) == 0
        or len(vertex_colors_rgb) == 0
    ):
        return RegionalAppearanceReport(
            palette_mode=pal_mode,
            palette_candidates_searched=pal_cnt,
            visible_skin_mean_delta_e00=None,
            occluded_core_mean_delta_e00=None,
            per_region_mean_delta_e00={},
            high_gradient_boundary_bricks_count=0,
            high_gradient_instance_ids=(),
            color_samples_available=False,
            sampled_skin_parts_count=0,
            sampled_core_parts_count=0,
        )

    v_pts = np.asarray(verts_ldu, dtype=np.float64)
    v_rgb = np.asarray(vertex_colors_rgb, dtype=np.float64)
    if float(np.max(v_rgb)) <= 1.0001:
        v_rgb = v_rgb * 255.0
    v_lab = srgb_to_cielab(v_rgb)
    reg_arr: Optional[np.ndarray] = None
    if vertex_region_labels is not None and len(vertex_region_labels) == v_pts.shape[0]:
        reg_arr = np.asarray([str(r) for r in vertex_region_labels], dtype=object)

    # Lookup Lab of any LDraw color code across both 54 solid and 322 LDConfig entries
    color_id_to_lab: Dict[int, np.ndarray] = {
        int(LDRAW_SOLID_PALETTE_SRGB[i][0]): LDRAW_SOLID_PALETTE_LAB[i]
        for i in range(len(LDRAW_SOLID_PALETTE_SRGB))
    }
    for rec in LDRAW_FULL_PALETTE_RECORDS:
        if rec.code not in color_id_to_lab:
            color_id_to_lab[int(rec.code)] = srgb_to_cielab(
                np.array([rec.rgb_u8], dtype=np.float64)
            )[0]
    default_lab = color_id_to_lab.get(71, np.array([67.5, -0.8, -2.5], dtype=np.float64))

    exposure_map = classify_part_surface_exposure(state, view_priority=view_priority)

    vis_des: List[float] = []
    vis_wts: List[float] = []
    occ_des: List[float] = []
    region_errs: Dict[str, List[float]] = {}
    high_grad_iids: List[int] = []

    for iid, pl in sorted(state.placed_parts.items()):
        xmin = min(b.xmin for b in pl.world_boxes) - 3.0
        xmax = max(b.xmax for b in pl.world_boxes) + 3.0
        ymin = min(b.ymin for b in pl.world_boxes) - 3.0
        ymax = max(b.ymax for b in pl.world_boxes) + 3.0
        zmin = min(b.zmin for b in pl.world_boxes) - 3.0
        zmax = max(b.zmax for b in pl.world_boxes) + 3.0

        inside = (
            (v_pts[:, 0] >= xmin)
            & (v_pts[:, 0] <= xmax)
            & (v_pts[:, 1] >= ymin)
            & (v_pts[:, 1] <= ymax)
            & (v_pts[:, 2] >= zmin)
            & (v_pts[:, 2] <= zmax)
        )
        if np.any(inside):
            sub_labs = v_lab[inside]
            sub_regs = reg_arr[inside] if reg_arr is not None else None
        else:
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            cz = 0.5 * (zmin + zmax)
            d2 = (v_pts[:, 0] - cx) ** 2 + (v_pts[:, 1] - cy) ** 2 + (v_pts[:, 2] - cz) ** 2
            k_nn = min(8, v_pts.shape[0])
            nn_idx = np.argpartition(d2, k_nn - 1)[:k_nn]
            sub_labs = v_lab[nn_idx]
            sub_regs = reg_arr[nn_idx] if reg_arr is not None else None

        p_lab = color_id_to_lab.get(int(pl.color_id), default_lab)
        sample_des = ciede2000_delta_e(sub_labs, np.repeat(p_lab[None, :], sub_labs.shape[0], axis=0))
        part_de = float(np.mean(sample_des))

        exp_info = exposure_map.get(iid)
        is_skin = not bool(exp_info.is_occluded_core) if exp_info is not None else True
        if is_skin:
            w_p = float(exp_info.mean_view_weight) if exp_info is not None else 1.0
            vis_des.append(part_de)
            vis_wts.append(w_p)
            if sub_labs.shape[0] >= 2:
                # Measure internal colour contrast across samples covered by this single exterior brick
                idx_lo = int(np.argmin(sub_labs[:, 0] + sub_labs[:, 1]))
                idx_hi = int(np.argmax(sub_labs[:, 0] + sub_labs[:, 1]))
                internal_span = float(
                    ciede2000_delta_e(sub_labs[idx_lo : idx_lo + 1], sub_labs[idx_hi : idx_hi + 1])[0]
                )
                if internal_span >= float(boundary_gradient_threshold_de00):
                    high_grad_iids.append(int(iid))
        else:
            occ_des.append(part_de)

        if sub_regs is not None:
            for r_lbl in set(sub_regs.tolist()):
                r_mask = sub_regs == r_lbl
                region_errs.setdefault(str(r_lbl), []).append(float(np.mean(sample_des[r_mask])))

    vis_mean: Optional[float] = (
        round(float(np.average(vis_des, weights=vis_wts)), 4)
        if vis_des and sum(vis_wts) > 0.0
        else None
    )
    occ_mean: Optional[float] = round(float(np.mean(occ_des)), 4) if occ_des else None
    per_reg = {k: round(float(np.mean(vs)), 4) for k, vs in sorted(region_errs.items())}
    return RegionalAppearanceReport(
        palette_mode=pal_mode,
        palette_candidates_searched=pal_cnt,
        visible_skin_mean_delta_e00=vis_mean,
        occluded_core_mean_delta_e00=occ_mean,
        per_region_mean_delta_e00=per_reg,
        high_gradient_boundary_bricks_count=len(high_grad_iids),
        high_gradient_instance_ids=tuple(high_grad_iids),
        color_samples_available=True,
        sampled_skin_parts_count=len(vis_des),
        sampled_core_parts_count=len(occ_des),
    )


def _reflect_box(box: LDUBox, axis: str, two_c: int) -> Tuple[int, int, int, int, int, int]:
    """Reflect `box` across `x = two_c / 2` or `y = two_c / 2` in exact integer LDU."""
    if axis == "x":
        return (two_c - box.xmax, box.ymin, box.zmin, two_c - box.xmin, box.ymax, box.zmax)
    return (box.xmin, two_c - box.ymax, box.zmin, box.xmax, two_c - box.ymin, box.zmax)


def _infer_part_region_labels(
    state: AssemblyState,
    verts_ldu: Optional[np.ndarray],
    vertex_region_labels: Optional[Sequence[Any]],
) -> Dict[int, str]:
    if (
        state.num_parts == 0
        or verts_ldu is None
        or vertex_region_labels is None
        or len(verts_ldu) == 0
        or len(vertex_region_labels) != len(verts_ldu)
    ):
        return {}
    v_pts = np.asarray(verts_ldu, dtype=np.float64)
    reg_arr = np.asarray([str(r) for r in vertex_region_labels], dtype=object)
    out: Dict[int, str] = {}
    for iid, pl in sorted(state.placed_parts.items()):
        xmin = min(b.xmin for b in pl.world_boxes) - 3.0
        xmax = max(b.xmax for b in pl.world_boxes) + 3.0
        ymin = min(b.ymin for b in pl.world_boxes) - 3.0
        ymax = max(b.ymax for b in pl.world_boxes) + 3.0
        zmin = min(b.zmin for b in pl.world_boxes) - 3.0
        zmax = max(b.zmax for b in pl.world_boxes) + 3.0
        inside = (
            (v_pts[:, 0] >= xmin)
            & (v_pts[:, 0] <= xmax)
            & (v_pts[:, 1] >= ymin)
            & (v_pts[:, 1] <= ymax)
            & (v_pts[:, 2] >= zmin)
            & (v_pts[:, 2] <= zmax)
        )
        if np.any(inside):
            vals = reg_arr[inside].tolist()
        else:
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            cz = 0.5 * (zmin + zmax)
            d2 = (v_pts[:, 0] - cx) ** 2 + (v_pts[:, 1] - cy) ** 2 + (v_pts[:, 2] - cz) ** 2
            nn_idx = int(np.argmin(d2))
            vals = [str(reg_arr[nn_idx])]
        counts: Dict[str, int] = {}
        for v in vals:
            counts[str(v)] = counts.get(str(v), 0) + 1
        out[int(iid)] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return out


def evaluate_bilateral_construction_symmetry(
    state: AssemblyState,
    axis: str = "x",
    plane_coord_ldu: Optional[float] = None,
    *,
    verts_ldu: Optional[np.ndarray] = None,
    vertex_region_labels: Optional[Sequence[Any]] = None,
    part_region_labels: Optional[Dict[int, str]] = None,
) -> BilateralSymmetryReport:
    """
    Evaluate bilateral brick-level construction symmetry vs. 2D/3D silhouette mirror overlap
    across `x = plane_coord_ldu` (`axis="x"`) or `y = plane_coord_ldu` (`axis="y"`).

    Distinguishes:
      - `volumetric_mirror_iou_5x5x2`: 3D `5x5x2 LDU` reflected volume overlap,
      - `orthographic_silhouette_mirror_iou` / `silhouette_mirror_iou`: 2D projected orthographic
        silhouette mirror IoU across plane-transverse camera views,
      - `exterior_construction_symmetry_ratio`: fraction of *exterior visible skin* non-centerline
        parts paired one-to-one with a reflected same-mold or opposite-hand counterpart (unaffected
        by completely enclosed internal core filler!),
      - `overall_construction_symmetry_ratio`: fraction across *all* non-centerline parts (skin + core),
      - `symmetric_color_agreement_ratio`: `None` when `comparable_pairs_count == 0`, else exact
        colour match fraction across paired parts.
    """
    from topo_brick.kernel.colored_inventory_allocator import classify_part_surface_exposure

    ax = str(axis).lower().strip()
    if ax not in ("x", "y"):
        raise ValueError(f"symmetry axis must be 'x' or 'y', got {axis!r}")

    if state.num_parts == 0:
        return BilateralSymmetryReport(
            symmetry_axis=ax,
            plane_coord_ldu=float(plane_coord_ldu or 0.0),
            total_parts=0,
            centerline_spanning_count=0,
            exact_same_mold_pairs=0,
            handed_counterpart_pairs=0,
            unmatched_parts_count=0,
            exterior_construction_symmetry_ratio=None,
            seam_alignment_ratio=None,
            silhouette_mirror_iou=0.0,
            symmetric_color_agreement_ratio=None,
            comparable_pairs_count=0,
            exterior_non_centerline_parts_count=0,
            exterior_paired_parts_count=0,
            exterior_unmatched_parts_count=0,
            overall_construction_symmetry_ratio=None,
            volumetric_mirror_iou_5x5x2=0.0,
            orthographic_silhouette_mirror_iou=0.0,
            per_region_symmetry_ratio={},
        )

    all_boxes = [b for p in state.placed_parts.values() for b in p.world_boxes]
    if plane_coord_ldu is None:
        if ax == "x":
            c_val = 0.5 * float(min(b.xmin for b in all_boxes) + max(b.xmax for b in all_boxes))
        else:
            c_val = 0.5 * float(min(b.ymin for b in all_boxes) + max(b.ymax for b in all_boxes))
    else:
        c_val = float(plane_coord_ldu)
    two_c = int(round(2.0 * c_val))

    # 1. Volumetric `5x5x2 LDU` & 2D orthographic silhouette mirror IoU around `c_val`
    xmin = min(min(b.xmin for b in all_boxes), min(two_c - b.xmax for b in all_boxes if ax == "x") if ax == "x" else min(b.xmin for b in all_boxes))
    xmax = max(max(b.xmax for b in all_boxes), max(two_c - b.xmin for b in all_boxes if ax == "x") if ax == "x" else max(b.xmax for b in all_boxes))
    ymin = min(min(b.ymin for b in all_boxes), min(two_c - b.ymax for b in all_boxes if ax == "y") if ax == "y" else min(b.ymin for b in all_boxes))
    ymax = max(max(b.ymax for b in all_boxes), max(two_c - b.ymin for b in all_boxes if ax == "y") if ax == "y" else max(b.ymax for b in all_boxes))
    zmin = min(b.zmin for b in all_boxes)
    zmax = max(b.zmax for b in all_boxes)

    gx = max(1, int(math.ceil((xmax - xmin) / 5.0)))
    gy = max(1, int(math.ceil((ymax - ymin) / 5.0)))
    gz = max(1, int(math.ceil((zmax - zmin) / 2.0)))
    occ_orig = np.zeros((gx, gy, gz), dtype=bool)
    occ_refl = np.zeros((gx, gy, gz), dtype=bool)
    for b in all_boxes:
        ox0 = max(0, (b.xmin - xmin) // 5)
        ox1 = min(gx, (b.xmax - xmin + 4) // 5)
        oy0 = max(0, (b.ymin - ymin) // 5)
        oy1 = min(gy, (b.ymax - ymin + 4) // 5)
        oz0 = max(0, (b.zmin - zmin) // 2)
        oz1 = min(gz, (b.zmax - zmin + 1) // 2)
        if ox1 > ox0 and oy1 > oy0 and oz1 > oz0:
            occ_orig[ox0:ox1, oy0:oy1, oz0:oz1] = True
        rb = _reflect_box(b, ax, two_c)
        rx0 = max(0, (rb[0] - xmin) // 5)
        rx1 = min(gx, (rb[3] - xmin + 4) // 5)
        ry0 = max(0, (rb[1] - ymin) // 5)
        ry1 = min(gy, (rb[4] - ymin + 4) // 5)
        if rx1 > rx0 and ry1 > ry0 and oz1 > oz0:
            occ_refl[rx0:rx1, ry0:ry1, oz0:oz1] = True
    vol_inter = int(np.count_nonzero(occ_orig & occ_refl))
    vol_union = int(np.count_nonzero(occ_orig | occ_refl))
    vol_iou = float(vol_inter / max(1, vol_union))

    def _proj_2d_iou(a2: np.ndarray, b2: np.ndarray) -> float:
        u2 = int(np.count_nonzero(a2 | b2))
        return float(np.count_nonzero(a2 & b2)) / float(max(1, u2))

    top_iou = _proj_2d_iou(np.any(occ_orig, axis=2), np.any(occ_refl, axis=2))
    elev_iou = (
        _proj_2d_iou(np.any(occ_orig, axis=1), np.any(occ_refl, axis=1))
        if ax == "x"
        else _proj_2d_iou(np.any(occ_orig, axis=0), np.any(occ_refl, axis=0))
    )
    ortho_sil_iou = 0.5 * (top_iou + elev_iou)

    # 2. Exterior visible skin vs. occluded internal core partition
    exposure_map = classify_part_surface_exposure(state)
    exterior_iids: Set[int] = {
        int(iid)
        for iid in state.placed_parts.keys()
        if not bool(exposure_map[iid].is_occluded_core) if iid in exposure_map
    }
    if not exposure_map:
        exterior_iids = set(int(iid) for iid in state.placed_parts.keys())

    # 3. Part-by-part outer seam AABB, reflected 5x5x2 occupancy & handed-counterpart correspondence
    part_aabb: Dict[int, Tuple[int, int, int, int, int, int]] = {}
    refl_aabb: Dict[int, Tuple[int, int, int, int, int, int]] = {}
    part_refl_occ_iou_fn_boxes: Dict[int, Tuple[LDUBox, ...]] = {}
    for iid, pl in sorted(state.placed_parts.items()):
        p_xmin = min(b.xmin for b in pl.world_boxes)
        p_ymin = min(b.ymin for b in pl.world_boxes)
        p_zmin = min(b.zmin for b in pl.world_boxes)
        p_xmax = max(b.xmax for b in pl.world_boxes)
        p_ymax = max(b.ymax for b in pl.world_boxes)
        p_zmax = max(b.zmax for b in pl.world_boxes)
        own_env = (p_xmin, p_ymin, p_zmin, p_xmax, p_ymax, p_zmax)
        ref_env = _reflect_box(LDUBox(*own_env), ax, two_c)
        part_aabb[iid] = own_env
        refl_aabb[iid] = ref_env
        part_refl_occ_iou_fn_boxes[iid] = pl.world_boxes

    def _aabb_close(
        a: Tuple[int, int, int, int, int, int],
        b: Tuple[int, int, int, int, int, int],
        tol: int = 2,
    ) -> bool:
        return all(abs(a[k] - b[k]) <= tol for k in range(6))

    def _reflected_part_iou(iid_a: int, iid_b: int) -> float:
        r_boxes_a = [LDUBox(*_reflect_box(bx, ax, two_c)) for bx in part_refl_occ_iou_fn_boxes[iid_a]]
        boxes_b = list(part_refl_occ_iou_fn_boxes[iid_b])
        ux0 = min(min(bx.xmin for bx in r_boxes_a), min(bx.xmin for bx in boxes_b))
        uy0 = min(min(bx.ymin for bx in r_boxes_a), min(bx.ymin for bx in boxes_b))
        uz0 = min(min(bx.zmin for bx in r_boxes_a), min(bx.zmin for bx in boxes_b))
        ux1 = max(max(bx.xmax for bx in r_boxes_a), max(bx.xmax for bx in boxes_b))
        uy1 = max(max(bx.ymax for bx in r_boxes_a), max(bx.ymax for bx in boxes_b))
        uz1 = max(max(bx.zmax for bx in r_boxes_a), max(bx.zmax for bx in boxes_b))
        nx_p = max(1, int(math.ceil((ux1 - ux0) / 5.0)))
        ny_p = max(1, int(math.ceil((uy1 - uy0) / 5.0)))
        nz_p = max(1, int(math.ceil((uz1 - uz0) / 2.0)))
        ga = np.zeros((nx_p, ny_p, nz_p), dtype=bool)
        gb = np.zeros((nx_p, ny_p, nz_p), dtype=bool)
        for bx in r_boxes_a:
            ga[
                max(0, (bx.xmin - ux0) // 5) : min(nx_p, (bx.xmax - ux0 + 4) // 5),
                max(0, (bx.ymin - uy0) // 5) : min(ny_p, (bx.ymax - uy0 + 4) // 5),
                max(0, (bx.zmin - uz0) // 2) : min(nz_p, (bx.zmax - uz0 + 1) // 2),
            ] = True
        for bx in boxes_b:
            gb[
                max(0, (bx.xmin - ux0) // 5) : min(nx_p, (bx.xmax - ux0 + 4) // 5),
                max(0, (bx.ymin - uy0) // 5) : min(ny_p, (bx.ymax - uy0 + 4) // 5),
                max(0, (bx.zmin - uz0) // 2) : min(nz_p, (bx.zmax - uz0 + 1) // 2),
            ] = True
        u_cnt = int(np.count_nonzero(ga | gb))
        return float(np.count_nonzero(ga & gb)) / float(max(1, u_cnt))

    centerline_iids: Set[int] = set()
    for iid in sorted(state.placed_parts.keys()):
        pl = state.placed_parts[iid]
        can_p = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pl.part_id) or pl.part_id
        if (
            _aabb_close(part_aabb[iid], refl_aabb[iid], tol=2)
            and can_p not in HANDED_COUNTERPART_MAP
            and _reflected_part_iou(iid, iid) >= 0.85
        ):
            centerline_iids.add(iid)

    def _pair_subset(cand_pool: Sequence[int]) -> Tuple[Set[int], int, int, int, int]:
        paired_loc: Set[int] = set()
        same_loc = 0
        handed_loc = 0
        col_loc = 0
        seam_loc = 0
        for iid in cand_pool:
            renv = refl_aabb[iid]
            cands = [oid for oid in cand_pool if oid != iid and _aabb_close(part_aabb[oid], renv, tol=2)]
            if cands:
                seam_loc += 1
            if iid in paired_loc:
                continue
            pl = state.placed_parts[iid]
            can_p = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pl.part_id) or pl.part_id
            handed_partner_pid = HANDED_COUNTERPART_MAP.get(can_p)
            matched_other: Optional[int] = None
            is_handed = False
            for oid in cands:
                if oid in paired_loc:
                    continue
                opl = state.placed_parts[oid]
                ocan_p = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(opl.part_id) or opl.part_id
                if ocan_p == can_p and ocan_p not in HANDED_COUNTERPART_MAP:
                    if _reflected_part_iou(iid, oid) >= 0.80:
                        matched_other = oid
                        is_handed = False
                        break
                elif handed_partner_pid is not None and ocan_p == handed_partner_pid:
                    if _reflected_part_iou(iid, oid) >= 0.80:
                        matched_other = oid
                        is_handed = True
                        break
            if matched_other is not None:
                paired_loc.add(iid)
                paired_loc.add(matched_other)
                if is_handed:
                    handed_loc += 1
                else:
                    same_loc += 1
                if int(pl.color_id) == int(state.placed_parts[matched_other].color_id):
                    col_loc += 1
        return paired_loc, same_loc, handed_loc, col_loc, seam_loc

    all_non_center = [iid for iid in sorted(state.placed_parts.keys()) if iid not in centerline_iids]
    ext_non_center = [iid for iid in all_non_center if iid in exterior_iids]
    ext_center_cnt = sum(1 for iid in centerline_iids if iid in exterior_iids)

    all_paired, same_all, handed_all, col_matches_all, _ = _pair_subset(all_non_center)
    ext_paired, same_ext, handed_ext, col_matches_ext, seam_ext = _pair_subset(ext_non_center)

    tot_pairs_all = same_all + handed_all
    unmatched_all = len(all_non_center) - 2 * tot_pairs_all
    overall_sym: Optional[float] = (
        (1.0 if len(centerline_iids) > 0 else None)
        if len(all_non_center) == 0
        else round(float(2 * tot_pairs_all) / float(len(all_non_center)), 6)
    )

    ext_pairs_tot = same_ext + handed_ext
    ext_paired_cnt = len(ext_paired)
    ext_unmatched_cnt = len(ext_non_center) - ext_paired_cnt
    if len(ext_non_center) == 0:
        ext_sym_ratio: Optional[float] = 1.0 if ext_center_cnt > 0 else None
        seam_ratio: Optional[float] = 1.0 if ext_center_cnt > 0 else None
    else:
        ext_sym_ratio = round(float(ext_paired_cnt) / float(len(ext_non_center)), 6)
        seam_ratio = round(float(seam_ext) / float(len(ext_non_center)), 6)

    # Honest `None` when zero comparable pairs exist!
    col_ratio: Optional[float] = (
        None if ext_pairs_tot == 0 else round(float(col_matches_ext) / float(ext_pairs_tot), 6)
    )

    eff_part_regs = (
        dict(part_region_labels)
        if part_region_labels is not None
        else _infer_part_region_labels(state, verts_ldu, vertex_region_labels)
    )
    per_reg_sym: Dict[str, Optional[float]] = {}
    if eff_part_regs:
        for r_lbl in sorted(set(eff_part_regs.values())):
            r_ext_nc = [iid for iid in ext_non_center if eff_part_regs.get(iid) == r_lbl]
            r_ext_c = [iid for iid in centerline_iids if iid in exterior_iids and eff_part_regs.get(iid) == r_lbl]
            if not r_ext_nc:
                per_reg_sym[r_lbl] = 1.0 if r_ext_c else None
            else:
                r_paired_cnt = sum(1 for iid in r_ext_nc if iid in ext_paired)
                per_reg_sym[r_lbl] = round(float(r_paired_cnt) / float(len(r_ext_nc)), 6)

    return BilateralSymmetryReport(
        symmetry_axis=ax,
        plane_coord_ldu=c_val,
        total_parts=int(state.num_parts),
        centerline_spanning_count=len(centerline_iids),
        exact_same_mold_pairs=same_ext,
        handed_counterpart_pairs=handed_ext,
        unmatched_parts_count=unmatched_all,
        exterior_construction_symmetry_ratio=ext_sym_ratio,
        seam_alignment_ratio=seam_ratio,
        silhouette_mirror_iou=round(ortho_sil_iou, 6),
        symmetric_color_agreement_ratio=col_ratio,
        comparable_pairs_count=ext_pairs_tot,
        exterior_non_centerline_parts_count=len(ext_non_center),
        exterior_paired_parts_count=ext_paired_cnt,
        exterior_unmatched_parts_count=ext_unmatched_cnt,
        overall_construction_symmetry_ratio=overall_sym,
        volumetric_mirror_iou_5x5x2=round(vol_iou, 6),
        orthographic_silhouette_mirror_iou=round(ortho_sil_iou, 6),
        per_region_symmetry_ratio=per_reg_sym,
    )
