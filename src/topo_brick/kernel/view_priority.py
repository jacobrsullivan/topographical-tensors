"""
Phase 11 (`Ext 5`): View-Ray Visibility Weighting (`ViewPrioritySpec`), `< 2.5 ms` Vectorized
Sub-Voxel Depth-Buffer & Silhouette-Rim Weighter, and 2D Camera Reprojection Evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree

from topo_brick.interface.port_manifold import AssemblyState


@dataclass(frozen=True, slots=True)
class ViewPrioritySpec:
    """
    Camera-viewpoint priority specification for single-image-to-brick synthesis (`Phase 11 Ext 5`).

    Prioritizes sculptural mold swaps and scarce color stock on camera-visible & silhouette-rim
    surfaces while keeping 3D 6-DoF static mechanics (`TABLE_FRICTION_ONLY` / `CLUTCHED_BASEPLATE`)
    strictly unweighted across all `360°`.
    """

    view_dir_ldu: Tuple[float, float, float]  # Direction from camera toward object in +Z-up LDU
    camera_pos_ldu: Optional[Tuple[float, float, float]] = None  # None = orthographic projection
    front_visible_weight: float = 2.5
    silhouette_rim_weight: float = 3.5
    occluded_rear_weight: float = 0.5


@dataclass(frozen=True, slots=True)
class CameraReprojectionMetrics:
    """2D camera-plane silhouette IoU and visible perceptual `CIEDE2000` error (`Phase 11 Ext 5`)."""

    silhouette_iou_2d: float
    visible_pixel_mean_delta_e00: float
    occluded_rear_mean_delta_e00: float
    foreground_pixel_count: int


def _camera_orthonormal_basis(
    view_dir_ldu: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right-handed orthonormal camera basis `(u_right, v_up, d_forward)` in Z-up LDU."""
    d = np.asarray(view_dir_ldu, dtype=np.float64)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9:
        d = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        d = d / dn
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(d, up_ref))) > 0.98:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(d, up_ref)
    u = u / max(1e-9, float(np.linalg.norm(u)))
    v = np.cross(u, d)
    v = v / max(1e-9, float(np.linalg.norm(v)))
    return u, v, d


def _project_points_to_uvd(
    pts_ldu: np.ndarray,
    spec: ViewPrioritySpec,
    u: np.ndarray,
    v: np.ndarray,
    d: np.ndarray,
    ref_center_ldu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project `(N, 3)` LDU points to `(u_coord, v_coord, depth)` in LDU units."""
    if spec.camera_pos_ldu is None:
        rel = pts_ldu - ref_center_ldu[None, :]
        u_c = rel @ u
        v_c = rel @ v
        dep = rel @ d
        return u_c, v_c, dep
    cam = np.asarray(spec.camera_pos_ldu, dtype=np.float64)
    focal_d = max(40.0, float(np.dot(ref_center_ldu - cam, d)))
    rays = pts_ldu - cam[None, :]
    dep = np.maximum(5.0, rays @ d)
    scale = focal_d / dep
    u_c = (rays @ u) * scale
    v_c = (rays @ v) * scale
    return u_c, v_c, dep


def compute_subvoxel_view_weights(
    occ_5x5x6: np.ndarray,
    spec: ViewPrioritySpec,
    pixel_pitch_ldu: float = 2.5,
) -> np.ndarray:
    """
    Compute a `(gx, gy, gz)` `float32` view-priority weight grid over `occ_5x5x6` in `< 2.5 ms`:
      - Occupied front-visible sub-voxels (`d <= d_min + 6 LDU`): `front_visible_weight` (`2.5`),
        upgraded to `silhouette_rim_weight` (`3.5`) on 2D contour rim pixels.
      - Occupied self-occluded rear sub-voxels: `occluded_rear_weight` (`0.5`).
      - Unoccupied exterior air sub-voxels jutting outside the 2D silhouette or in front of the
        visible surface: `silhouette_rim_weight` (`3.5`) / `front_visible_weight` (`2.5`), while
        self-occluded rear air receives `occluded_rear_weight` (`0.5`).
    """
    gx, gy, gz = occ_5x5x6.shape
    if not np.any(occ_5x5x6):
        return np.ones((gx, gy, gz), dtype=np.float32)

    u_ax, v_ax, d_ax = _camera_orthonormal_basis(spec.view_dir_ldu)

    # Full grid center coordinates
    xs = (np.arange(gx, dtype=np.float64) + 0.5) * 5.0
    ys = (np.arange(gy, dtype=np.float64) + 0.5) * 5.0
    zs = (np.arange(gz, dtype=np.float64) + 0.5) * 6.0

    occ_idx = np.argwhere(occ_5x5x6)
    occ_pts = np.column_stack([xs[occ_idx[:, 0]], ys[occ_idx[:, 1]], zs[occ_idx[:, 2]]])
    ref_center = np.mean(occ_pts, axis=0)

    # Project all grid sub-voxels in one vectorized matmul
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
    all_pts = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    u_all, v_all, d_all = _project_points_to_uvd(all_pts, spec, u_ax, v_ax, d_ax, ref_center)

    u_min = float(np.min(u_all)) - pixel_pitch_ldu
    v_min = float(np.min(v_all)) - pixel_pitch_ldu
    iu_all = np.floor((u_all - u_min) / pixel_pitch_ldu).astype(np.int32) + 1
    iv_all = np.floor((v_all - v_min) / pixel_pitch_ldu).astype(np.int32) + 1
    nu = int(np.max(iu_all)) + 3
    nv = int(np.max(iv_all)) + 3

    occ_flat = occ_5x5x6.ravel()
    d_min_buf = np.full((nu, nv), np.inf, dtype=np.float64)
    np.minimum.at(d_min_buf, (iu_all[occ_flat], iv_all[occ_flat]), d_all[occ_flat])

    fg_mask = np.isfinite(d_min_buf)
    # 4-connected erosion to detect 2D silhouette rim pixels
    eroded = np.zeros_like(fg_mask)
    eroded[1:-1, 1:-1] = (
        fg_mask[1:-1, 1:-1]
        & fg_mask[2:, 1:-1]
        & fg_mask[:-2, 1:-1]
        & fg_mask[1:-1, 2:]
        & fg_mask[1:-1, :-2]
    )
    rim_mask = fg_mask & (~eroded)

    pix_dmin = d_min_buf[iu_all, iv_all]
    pix_fg = fg_mask[iu_all, iv_all]
    pix_rim = rim_mask[iu_all, iv_all]

    w_front = float(spec.front_visible_weight)
    w_rim = float(spec.silhouette_rim_weight)
    w_rear = float(spec.occluded_rear_weight)

    weights_flat = np.empty(all_pts.shape[0], dtype=np.float32)
    # Occupied sub-voxels
    vis_occ = occ_flat & (d_all <= pix_dmin + 6.0)
    rim_occ = vis_occ & pix_rim
    front_occ = vis_occ & (~pix_rim)
    rear_occ = occ_flat & (~vis_occ)
    weights_flat[rim_occ] = w_rim
    weights_flat[front_occ] = w_front
    weights_flat[rear_occ] = w_rear

    # Unoccupied air sub-voxels (overfill spill weights):
    # Under perspective cameras (`camera_pos_ldu is not None`), also check orthographic lateral bounds so
    # radially diverging perspective rays behind the front surface never misclassify exterior side air as occluded!
    if spec.camera_pos_ldu is not None:
        rel_o = all_pts - ref_center[None, :]
        uo_all = rel_o @ u_ax
        vo_all = rel_o @ v_ax
        uo_min, uo_max = float(np.min(uo_all[occ_flat])), float(np.max(uo_all[occ_flat]))
        vo_min, vo_max = float(np.min(vo_all[occ_flat])), float(np.max(vo_all[occ_flat]))
        inside_ortho_box = (
            (uo_all >= uo_min - 2.5)
            & (uo_all <= uo_max + 2.5)
            & (vo_all >= vo_min - 2.5)
            & (vo_all <= vo_max + 2.5)
        )
    else:
        inside_ortho_box = np.ones(all_pts.shape[0], dtype=bool)

    air_mask = ~occ_flat
    air_occluded_behind = air_mask & pix_fg & inside_ortho_box & (d_all > pix_dmin + 2.5)
    air_in_front = air_mask & pix_fg & (~air_occluded_behind)
    air_bg_sil = air_mask & (~pix_fg)
    weights_flat[air_bg_sil] = 1.6 * w_rim
    weights_flat[air_in_front] = w_front
    weights_flat[air_occluded_behind] = max(0.15, 0.5 * w_rear)

    return weights_flat.reshape(gx, gy, gz)


def compute_part_camera_visibility_weights(
    state: AssemblyState,
    spec: ViewPrioritySpec,
    pixel_pitch_ldu: float = 2.5,
) -> Dict[int, float]:
    """
    Compute exact per-part camera visibility priority weights in `< 1.2 ms` by projecting `state`'s
    built `5x5x6 LDU` sub-voxels onto the camera depth buffer (`np.minimum.at`):
      - Self-occluded rear parts (`0` front-visible pixels) receive `0.08 * spec.occluded_rear_weight`.
      - Camera-visible front parts receive `spec.front_visible_weight + rim_bonus + pixel_area_bonus`,
        creating a sharp priority gradient so TU min-cost chromatic flow routes scarce matching colors
        exclusively to camera-facing pixels first.
    """
    if state.num_parts == 0:
        return {}
    all_boxes = [b for pl in state.placed_parts.values() for b in pl.world_boxes]
    if not all_boxes:
        return {iid: 1.0 for iid in state.placed_parts}

    xmin = min(b.xmin for b in all_boxes)
    ymin = min(b.ymin for b in all_boxes)
    zmin = min(b.zmin for b in all_boxes)
    xmax = max(b.xmax for b in all_boxes)
    ymax = max(b.ymax for b in all_boxes)
    zmax = max(b.zmax for b in all_boxes)

    ox = int(math.floor(xmin / 5.0)) * 5
    oy = int(math.floor(ymin / 5.0)) * 5
    oz = int(math.floor(zmin / 6.0)) * 6
    gx = max(1, int(math.ceil((xmax - ox) / 5.0)))
    gy = max(1, int(math.ceil((ymax - oy) / 5.0)))
    gz = max(1, int(math.ceil((zmax - oz) / 6.0)))

    inst_grid = np.full((gx, gy, gz), -1, dtype=np.int32)
    for iid, pl in sorted(state.placed_parts.items()):
        for b in pl.world_boxes:
            x0 = max(0, int(math.floor((b.xmin - ox) / 5.0)))
            x1 = min(gx, int(math.ceil((b.xmax - ox) / 5.0)))
            y0 = max(0, int(math.floor((b.ymin - oy) / 5.0)))
            y1 = min(gy, int(math.ceil((b.ymax - oy) / 5.0)))
            z0 = max(0, int(math.floor((b.zmin - oz) / 6.0)))
            z1 = min(gz, int(math.ceil((b.zmax - oz) / 6.0)))
            if x1 > x0 and y1 > y0 and z1 > z0:
                inst_grid[x0:x1, y0:y1, z0:z1] = int(iid)

    b_idx = np.argwhere(inst_grid >= 0)
    if b_idx.shape[0] == 0:
        return {iid: 1.0 for iid in state.placed_parts}

    xs = ox + (np.arange(gx, dtype=np.float64) + 0.5) * 5.0
    ys = oy + (np.arange(gy, dtype=np.float64) + 0.5) * 5.0
    zs = oz + (np.arange(gz, dtype=np.float64) + 0.5) * 6.0
    b_pts = np.column_stack([xs[b_idx[:, 0]], ys[b_idx[:, 1]], zs[b_idx[:, 2]]])
    b_iids = inst_grid[b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]]

    u_ax, v_ax, d_ax = _camera_orthonormal_basis(spec.view_dir_ldu)
    ref_center = np.mean(b_pts, axis=0)
    ub, vb, db = _project_points_to_uvd(b_pts, spec, u_ax, v_ax, d_ax, ref_center)

    u_min = float(np.min(ub)) - pixel_pitch_ldu
    v_min = float(np.min(vb)) - pixel_pitch_ldu
    iub = np.floor((ub - u_min) / pixel_pitch_ldu).astype(np.int32) + 1
    ivb = np.floor((vb - v_min) / pixel_pitch_ldu).astype(np.int32) + 1
    nu = int(np.max(iub)) + 3
    nv = int(np.max(ivb)) + 3

    d_min_buf = np.full((nu, nv), np.inf, dtype=np.float64)
    np.minimum.at(d_min_buf, (iub, ivb), db)

    fg_mask = np.isfinite(d_min_buf)
    eroded = np.zeros_like(fg_mask)
    eroded[1:-1, 1:-1] = (
        fg_mask[1:-1, 1:-1]
        & fg_mask[2:, 1:-1]
        & fg_mask[:-2, 1:-1]
        & fg_mask[1:-1, 2:]
        & fg_mask[1:-1, :-2]
    )
    rim_mask = fg_mask & (~eroded)

    # Exact front-most winner per pixel (`db <= d_min_buf + 2.5 LDU`)
    is_vis = db <= (d_min_buf[iub, ivb] + 2.5)
    is_rim = is_vis & rim_mask[iub, ivb]

    max_iid = max(state.placed_parts.keys()) + 1
    tot_cnt = np.bincount(b_iids, minlength=max_iid)
    vis_cnt = np.bincount(b_iids[is_vis], minlength=max_iid)
    rim_cnt = np.bincount(b_iids[is_rim], minlength=max_iid)
    max_vis_cnt = max(1, int(np.max(vis_cnt)))

    out: Dict[int, float] = {}
    for iid in state.placed_parts:
        vc = int(vis_cnt[iid])
        if vc == 0:
            out[iid] = max(0.02, 0.08 * float(spec.occluded_rear_weight))
        else:
            rc = int(rim_cnt[iid])
            tc = max(1, int(tot_cnt[iid]))
            vis_ratio = float(vc) / float(tc)
            area_share = float(vc) / float(max_vis_cnt)
            rim_share = float(rc) / float(vc)
            out[iid] = (
                float(spec.front_visible_weight) * (0.65 + 0.85 * area_share + 0.35 * vis_ratio)
                + (float(spec.silhouette_rim_weight) - float(spec.front_visible_weight)) * rim_share
            )
    return out


def evaluate_camera_reprojection_fidelity(
    state: AssemblyState,
    occ_5x5x6: np.ndarray,
    spec: ViewPrioritySpec,
    verts_ldu: Optional[np.ndarray] = None,
    vertex_colors_rgb: Optional[np.ndarray] = None,
    pixel_pitch_ldu: float = 2.5,
) -> CameraReprojectionMetrics:
    """
    Evaluate 2D camera-plane `silhouette_iou_2d` and visible-pixel perceptual `CIEDE2000` (`ΔE_00`)
    of `state` against target `occ_5x5x6` and `(verts_ldu, vertex_colors_rgb)` in `< 4 ms`.
    """
    from topo_brick.kernel.open_catalog_realizer import (
        LDRAW_SOLID_PALETTE_LAB,
        LDRAW_SOLID_PALETTE_SRGB,
        ciede2000_delta_e,
        srgb_to_cielab,
    )

    gx, gy, gz = occ_5x5x6.shape
    if not np.any(occ_5x5x6):
        return CameraReprojectionMetrics(1.0, 0.0, 0.0, 0)

    # Rasterize built `world_boxes` onto `(gx, gy, gz)` `5x5x6 LDU` grid with `instance_id` tracking
    built_occ = np.zeros((gx, gy, gz), dtype=bool)
    built_inst = np.full((gx, gy, gz), -1, dtype=np.int32)
    for iid, pl in sorted(state.placed_parts.items()):
        for b in pl.world_boxes:
            x0 = max(0, int(math.floor(b.xmin / 5.0)))
            x1 = min(gx, int(math.ceil(b.xmax / 5.0)))
            y0 = max(0, int(math.floor(b.ymin / 5.0)))
            y1 = min(gy, int(math.ceil(b.ymax / 5.0)))
            z0 = max(0, int(math.floor(b.zmin / 6.0)))
            z1 = min(gz, int(math.ceil(b.zmax / 6.0)))
            if x1 > x0 and y1 > y0 and z1 > z0:
                built_occ[x0:x1, y0:y1, z0:z1] = True
                built_inst[x0:x1, y0:y1, z0:z1] = int(iid)

    u_ax, v_ax, d_ax = _camera_orthonormal_basis(spec.view_dir_ldu)
    xs = (np.arange(gx, dtype=np.float64) + 0.5) * 5.0
    ys = (np.arange(gy, dtype=np.float64) + 0.5) * 5.0
    zs = (np.arange(gz, dtype=np.float64) + 0.5) * 6.0

    t_idx = np.argwhere(occ_5x5x6)
    t_pts = np.column_stack([xs[t_idx[:, 0]], ys[t_idx[:, 1]], zs[t_idx[:, 2]]])
    ref_center = np.mean(t_pts, axis=0)

    b_idx = np.argwhere(built_occ)
    b_pts = (
        np.column_stack([xs[b_idx[:, 0]], ys[b_idx[:, 1]], zs[b_idx[:, 2]]])
        if b_idx.shape[0] > 0
        else np.zeros((0, 3), dtype=np.float64)
    )

    ut, vt, dt = _project_points_to_uvd(t_pts, spec, u_ax, v_ax, d_ax, ref_center)
    if b_pts.shape[0] > 0:
        ub, vb, db = _project_points_to_uvd(b_pts, spec, u_ax, v_ax, d_ax, ref_center)
        u_min = min(float(np.min(ut)), float(np.min(ub))) - pixel_pitch_ldu
        v_min = min(float(np.min(vt)), float(np.min(vb))) - pixel_pitch_ldu
        u_max = max(float(np.max(ut)), float(np.max(ub))) + pixel_pitch_ldu
        v_max = max(float(np.max(vt)), float(np.max(vb))) + pixel_pitch_ldu
    else:
        ub = vb = db = np.zeros(0, dtype=np.float64)
        u_min = float(np.min(ut)) - pixel_pitch_ldu
        v_min = float(np.min(vt)) - pixel_pitch_ldu
        u_max = float(np.max(ut)) + pixel_pitch_ldu
        v_max = float(np.max(vt)) + pixel_pitch_ldu

    nu = max(4, int(math.ceil((u_max - u_min) / pixel_pitch_ldu)) + 2)
    nv = max(4, int(math.ceil((v_max - v_min) / pixel_pitch_ldu)) + 2)

    iut = np.clip(np.floor((ut - u_min) / pixel_pitch_ldu).astype(np.int32), 0, nu - 1)
    ivt = np.clip(np.floor((vt - v_min) / pixel_pitch_ldu).astype(np.int32), 0, nv - 1)
    tgt_sil = np.zeros((nu, nv), dtype=bool)
    tgt_sil[iut, ivt] = True

    blt_sil = np.zeros((nu, nv), dtype=bool)
    blt_dmin = np.full((nu, nv), np.inf, dtype=np.float64)
    blt_dmax = np.full((nu, nv), -np.inf, dtype=np.float64)
    blt_front_iid = np.full((nu, nv), -1, dtype=np.int32)
    blt_rear_iid = np.full((nu, nv), -1, dtype=np.int32)

    if b_pts.shape[0] > 0:
        iub = np.clip(np.floor((ub - u_min) / pixel_pitch_ldu).astype(np.int32), 0, nu - 1)
        ivb = np.clip(np.floor((vb - v_min) / pixel_pitch_ldu).astype(np.int32), 0, nv - 1)
        blt_sil[iub, ivb] = True
        b_iids = built_inst[b_idx[:, 0], b_idx[:, 1], b_idx[:, 2]]
        np.minimum.at(blt_dmin, (iub, ivb), db)
        np.maximum.at(blt_dmax, (iub, ivb), db)
        is_f = db <= (blt_dmin[iub, ivb] + 1e-6)
        blt_front_iid[iub[is_f], ivb[is_f]] = b_iids[is_f]
        is_r = db >= (blt_dmax[iub, ivb] - 1e-6)
        blt_rear_iid[iub[is_r], ivb[is_r]] = b_iids[is_r]

    # Dilate 1 pixel (3x3 box) to close sub-voxel point-splat pinholes before 2D silhouette IoU
    def _close3(m: np.ndarray) -> np.ndarray:
        p = np.pad(m, 1, mode="constant", constant_values=False)
        dil = (
            p[1:-1, 1:-1] | p[2:, 1:-1] | p[:-2, 1:-1] | p[1:-1, 2:] | p[1:-1, :-2]
            | p[2:, 2:] | p[2:, :-2] | p[:-2, 2:] | p[:-2, :-2]
        )
        p2 = np.pad(dil, 1, mode="constant", constant_values=False)
        return (
            p2[1:-1, 1:-1] & p2[2:, 1:-1] & p2[:-2, 1:-1] & p2[1:-1, 2:] & p2[1:-1, :-2]
        )

    t_s_closed = _close3(tgt_sil)
    b_s_closed = _close3(blt_sil)
    inter_2d = int(np.count_nonzero(t_s_closed & b_s_closed))
    union_2d = int(np.count_nonzero(t_s_closed | b_s_closed))
    sil_iou = float(inter_2d) / float(max(1, union_2d))

    vis_de = 0.0
    rear_de = 0.0
    if (
        verts_ldu is not None
        and vertex_colors_rgb is not None
        and verts_ldu.shape[0] > 0
        and vertex_colors_rgb.shape[0] > 0
        and b_pts.shape[0] > 0
    ):
        v_pts = np.asarray(verts_ldu, dtype=np.float64)
        v_rgb = np.asarray(vertex_colors_rgb, dtype=np.float64)
        if float(np.max(v_rgb)) <= 1.0001:
            v_rgb = v_rgb * 255.0
        v_lab = srgb_to_cielab(v_rgb)
        uv_m, vv_m, dv_m = _project_points_to_uvd(v_pts, spec, u_ax, v_ax, d_ax, ref_center)

        # Split target vertex samples into front-facing vs rear-facing halves in camera depth
        d_med = float(np.median(dv_m))
        front_m = dv_m <= d_med + 12.0
        rear_m = dv_m >= d_med - 12.0
        tree_front = cKDTree(np.column_stack([uv_m[front_m], vv_m[front_m]])) if np.any(front_m) else cKDTree(np.column_stack([uv_m, vv_m]))
        labs_front_src = v_lab[front_m] if np.any(front_m) else v_lab
        tree_rear = cKDTree(np.column_stack([uv_m[rear_m], vv_m[rear_m]])) if np.any(rear_m) else cKDTree(np.column_stack([uv_m, vv_m]))
        labs_rear_src = v_lab[rear_m] if np.any(rear_m) else v_lab

        codes = [c[0] for c in LDRAW_SOLID_PALETTE_SRGB]
        id_to_lab = {int(codes[i]): LDRAW_SOLID_PALETTE_LAB[i] for i in range(len(codes))}
        default_lab = id_to_lab.get(71, np.array([67.5, -0.8, -2.5], dtype=np.float64))

        fg_pix = np.argwhere(blt_front_iid >= 0)
        if fg_pix.shape[0] > 0:
            pix_uv = np.column_stack(
                [
                    u_min + (fg_pix[:, 0].astype(np.float64) + 0.5) * pixel_pitch_ldu,
                    v_min + (fg_pix[:, 1].astype(np.float64) + 0.5) * pixel_pitch_ldu,
                ]
            )
            _, nn_f = tree_front.query(pix_uv, k=1)
            f_iids = blt_front_iid[fg_pix[:, 0], fg_pix[:, 1]]
            b_labs = np.stack(
                [id_to_lab.get(state.placed_parts[int(ii)].color_id, default_lab) for ii in f_iids],
                axis=0,
            )
            t_labs = labs_front_src[np.asarray(nn_f, dtype=np.int64)]
            vis_de = float(np.mean(ciede2000_delta_e(b_labs, t_labs)))

        rear_pix = np.argwhere((blt_rear_iid >= 0) & (blt_rear_iid != blt_front_iid))
        if rear_pix.shape[0] > 0:
            pix_uv_r = np.column_stack(
                [
                    u_min + (rear_pix[:, 0].astype(np.float64) + 0.5) * pixel_pitch_ldu,
                    v_min + (rear_pix[:, 1].astype(np.float64) + 0.5) * pixel_pitch_ldu,
                ]
            )
            _, nn_r = tree_rear.query(pix_uv_r, k=1)
            r_iids = blt_rear_iid[rear_pix[:, 0], rear_pix[:, 1]]
            br_labs = np.stack(
                [id_to_lab.get(state.placed_parts[int(ii)].color_id, default_lab) for ii in r_iids],
                axis=0,
            )
            tr_labs = labs_rear_src[np.asarray(nn_r, dtype=np.int64)]
            rear_de = float(np.mean(ciede2000_delta_e(br_labs, tr_labs)))

    return CameraReprojectionMetrics(
        silhouette_iou_2d=round(sil_iou, 6),
        visible_pixel_mean_delta_e00=round(vis_de, 4),
        occluded_rear_mean_delta_e00=round(rear_de, 4),
        foreground_pixel_count=int(np.count_nonzero(t_s_closed)),
    )
