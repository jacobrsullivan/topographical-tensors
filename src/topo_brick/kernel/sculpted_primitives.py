"""
Shared palettes, `CIEDE2000` perceptual color quantization, `2 LDU` CSG voxel caches,
relative `5x5x6` masks, and boolean course-tiling primitives (`Phase 15`).
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.interface.brick_catalog import (
    ConnectorType,
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame, rot_id_to_matrix
from topo_brick.interface.port_manifold import AssemblyState, get_admissible_socket_pocket_ldu
from topo_brick.representation.interface_fields import SparseTiledTargetField


# Zero-cost Option 3B terminal `8 LDU` plate -> smooth `8 LDU` tile equivalence map
PLATE_TO_SMOOTH_TILE_MAP: Dict[str, str] = {
    "3024": "3070b",  # 1x1
    "3023": "3069b",  # 1x2
    "3623": "63864",  # 1x3
    "3710": "2431",   # 1x4
    "3666": "6636",   # 1x6
    "3460": "4162",   # 1x8
    "3022": "3068b",  # 2x2
    "3021": "26603",  # 2x3
    "3020": "87079",  # 2x4
    "3795": "69729",  # 2x6
}

# Zero-IoU-delta `24 LDU` brick -> `(8 LDU plate, 8 LDU smooth tile)` footprint equivalence map
BRICK_TO_PLATE_AND_TILE_MAP: Dict[str, Tuple[str, str]] = {
    "3005": ("3024", "3070b"),  # 1x1
    "3004": ("3023", "3069b"),  # 1x2
    "3622": ("3623", "63864"),  # 1x3
    "3010": ("3710", "2431"),   # 1x4
    "3009": ("3666", "6636"),   # 1x6
    "3008": ("3460", "4162"),   # 1x8
    "3003": ("3022", "3068b"),  # 2x2
    "3002": ("3021", "26603"),  # 2x3
    "3001": ("3020", "87079"),  # 2x4
    "2456": ("3795", "69729"),  # 2x6
}

# Official LDraw Solid LEGO Palette: `(ldraw_color_id, name, (R_u8, G_u8, B_u8))`
LDRAW_SOLID_PALETTE_SRGB: Tuple[Tuple[int, str, Tuple[int, int, int]], ...] = (
    (0, "Black", (27, 42, 52)),
    (1, "Blue", (0, 85, 191)),
    (2, "Green", (35, 120, 65)),
    (3, "Dark_Turquoise", (0, 143, 155)),
    (4, "Red", (201, 26, 9)),
    (5, "Dark_Pink", (200, 112, 160)),
    (6, "Brown", (88, 57, 39)),
    (7, "Light_Gray", (155, 161, 157)),
    (8, "Dark_Gray", (109, 110, 92)),
    (9, "Light_Blue", (180, 210, 227)),
    (10, "Bright_Green", (75, 159, 74)),
    (11, "Light_Turquoise", (85, 165, 175)),
    (12, "Salmon", (242, 112, 94)),
    (13, "Pink", (252, 151, 172)),
    (14, "Yellow", (242, 205, 55)),
    (15, "White", (244, 244, 244)),
    (17, "Light_Green", (194, 218, 184)),
    (18, "Light_Yellow", (251, 230, 150)),
    (19, "Tan", (228, 205, 158)),
    (20, "Light_Violet", (201, 202, 226)),
    (22, "Purple", (129, 0, 123)),
    (25, "Orange", (254, 138, 24)),
    (26, "Magenta", (146, 57, 120)),
    (27, "Lime", (187, 233, 11)),
    (28, "Dark_Tan", (149, 138, 115)),
    (29, "Bright_Pink", (228, 173, 200)),
    (30, "Medium_Lavender", (172, 120, 186)),
    (31, "Lavender", (225, 213, 237)),
    (68, "Very_Light_Orange", (243, 207, 155)),
    (70, "Reddish_Brown", (88, 42, 18)),
    (71, "Light_Bluish_Gray", (160, 165, 169)),
    (72, "Dark_Bluish_Gray", (108, 110, 104)),
    (73, "Medium_Blue", (90, 147, 219)),
    (74, "Medium_Green", (115, 220, 161)),
    (77, "Light_Pink", (254, 204, 207)),
    (78, "Light_Nougat", (246, 215, 179)),
    (84, "Medium_Nougat", (170, 125, 85)),
    (85, "Dark_Purple", (63, 54, 145)),
    (191, "Bright_Light_Orange", (248, 187, 61)),
    (212, "Bright_Light_Blue", (159, 195, 233)),
    (226, "Bright_Light_Yellow", (255, 240, 58)),
    (272, "Dark_Blue", (10, 52, 99)),
    (288, "Dark_Green", (24, 70, 50)),
    (308, "Dark_Brown", (53, 33, 0)),
    (320, "Dark_Red", (114, 14, 15)),
    (321, "Dark_Azur", (7, 139, 201)),
    (322, "Medium_Azur", (54, 174, 191)),
    (323, "Light_Aqua", (173, 221, 208)),
    (326, "Yellowish_Green", (223, 238, 165)),
    (330, "Olive_Green", (119, 119, 78)),
    (353, "Coral", (255, 105, 143)),
    (378, "Sand_Green", (160, 188, 172)),
    (379, "Sand_Blue", (96, 116, 161)),
    (484, "Dark_Orange", (169, 85, 0)),
)


def srgb_to_cielab(rgb_0_255: np.ndarray) -> np.ndarray:
    """Convert `(..., 3)` sRGB `[0..255]` (or `[0..1]` if max <= 1.0) to CIE `L*a*b*` (D65)."""
    arr = np.asarray(rgb_0_255, dtype=np.float64)
    if arr.size > 0 and float(np.max(arr)) <= 1.0001:
        arr = arr * 255.0
    v = np.clip(arr / 255.0, 0.0, 1.0)
    lin = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    # sRGB linear to XYZ (D65)
    m_xyz = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ m_xyz.T
    ref = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)
    xr = xyz / ref
    eps = 216.0 / 24389.0
    kap = 24389.0 / 27.0
    f = np.where(xr > eps, np.cbrt(np.maximum(xr, 0.0)), (kap * xr + 16.0) / 116.0)
    l_star = 116.0 * f[..., 1] - 16.0
    a_star = 500.0 * (f[..., 0] - f[..., 1])
    b_star = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l_star, a_star, b_star], axis=-1)


def ciede2000_delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized ISO/CIE 11664-6 `CIEDE2000` (`ΔE_00`) perceptual color difference."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7)))

    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dl_p = l2 - l1
    dc_p = c2p - c1p

    c_prod = c1p * c2p
    dh_deg = h2p - h1p
    dh_deg = np.where(dh_deg > 180.0, dh_deg - 360.0, dh_deg)
    dh_deg = np.where(dh_deg < -180.0, dh_deg + 360.0, dh_deg)
    dh_deg = np.where(c_prod == 0.0, 0.0, dh_deg)
    dh_p = 2.0 * np.sqrt(np.maximum(c_prod, 0.0)) * np.sin(np.radians(0.5 * dh_deg))

    l_bar_p = 0.5 * (l1 + l2)
    c_bar_p = 0.5 * (c1p + c2p)

    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    h_bar_p = np.where(
        c_prod == 0.0,
        h_sum,
        np.where(
            h_diff <= 180.0,
            0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    t_fac = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    c_bar_p7 = c_bar_p**7
    r_c = 2.0 * np.sqrt(c_bar_p7 / (c_bar_p7 + 25.0**7))
    l_m50_sq = (l_bar_p - 50.0) ** 2
    s_l = 1.0 + (0.015 * l_m50_sq) / np.sqrt(20.0 + l_m50_sq)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t_fac
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c

    t_l = dl_p / s_l
    t_c = dc_p / s_c
    t_h = dh_p / s_h
    return np.sqrt(np.maximum(0.0, t_l * t_l + t_c * t_c + t_h * t_h + r_t * t_c * t_h))


_PALETTE_CODES: np.ndarray = np.array([c[0] for c in LDRAW_SOLID_PALETTE_SRGB], dtype=np.int32)
_PALETTE_SRGB: np.ndarray = np.array([c[2] for c in LDRAW_SOLID_PALETTE_SRGB], dtype=np.float64)
LDRAW_SOLID_PALETTE_LAB: np.ndarray = srgb_to_cielab(_PALETTE_SRGB)


def assign_perceptual_ldraw_colors(
    state: AssemblyState,
    verts_ldu: np.ndarray,
    vertex_colors_rgb: np.ndarray,
    *,
    open_color_palette_mode: str = "solid_54_default",
    view_priority: Optional[Any] = None,
    preferred_finish: Optional[str] = None,
) -> float:
    """
    Assign official LDraw `color_id` to every placed part in `state` by sampling mesh vertices
    `(verts_ldu, vertex_colors_rgb)` and matching `LDRAW_SOLID_PALETTE_LAB` (or finish-penalized
    `LDConfig.ldr` in `ldconfig_finish_aware` mode) via `CIEDE2000` (`ΔE_00`). When `view_priority`
    is supplied, weights per-sample `ΔE_00` toward camera-facing samples along `-view_dir_ldu`.
    """
    if state.num_parts == 0 or verts_ldu.shape[0] == 0 or vertex_colors_rgb.shape[0] == 0:
        return 0.0
    v_pts = np.asarray(verts_ldu, dtype=np.float64)
    v_rgb = np.asarray(vertex_colors_rgb, dtype=np.float64)
    if float(np.max(v_rgb)) <= 1.0001:
        v_rgb = v_rgb * 255.0

    # Subsample up to 2,000 vertices deterministically for `< 1.5 ms` color lookup
    if v_pts.shape[0] > 2000:
        step = max(1, v_pts.shape[0] // 2000)
        v_pts = v_pts[::step]
        v_rgb = v_rgb[::step]
    v_lab = srgb_to_cielab(v_rgb)

    cam_toward_vec: Optional[np.ndarray] = None
    w_front_cfg = 2.5
    w_rear_cfg = 0.5
    if view_priority is not None and hasattr(view_priority, "view_dir_ldu"):
        vd = np.asarray(view_priority.view_dir_ldu, dtype=np.float64)
        vn = float(np.linalg.norm(vd))
        if vn > 1.0e-9:
            cam_toward_vec = -(vd / vn)
        w_front_cfg = max(1.0, float(getattr(view_priority, "front_visible_weight", 2.5)))
        w_rear_cfg = max(0.05, float(getattr(view_priority, "occluded_rear_weight", 0.5)))

    if open_color_palette_mode == "ldconfig_finish_aware":
        cand_codes, cand_labs, finish_penalties = _get_ldconfig_full_palette_lab_and_penalties(
            preferred_finish=preferred_finish
        )
    else:
        cand_codes = _PALETTE_CODES
        cand_labs = LDRAW_SOLID_PALETTE_LAB
        finish_penalties = np.zeros(cand_labs.shape[0], dtype=np.float64)

    delta_es: List[float] = []
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
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

        if np.any(inside):
            sub_pts = v_pts[inside]
            sub_labs = v_lab[inside]
        else:
            d2 = (v_pts[:, 0] - cx) ** 2 + (v_pts[:, 1] - cy) ** 2 + (v_pts[:, 2] - cz) ** 2
            k_nn = min(8, v_pts.shape[0])
            nn_idx = np.argpartition(d2, k_nn - 1)[:k_nn]
            sub_pts = v_pts[nn_idx]
            sub_labs = v_lab[nn_idx]

        if cam_toward_vec is None:
            target_lab = np.mean(sub_labs, axis=0)
            raw_de = ciede2000_delta_e(target_lab[None, :], cand_labs)
        else:
            if sub_pts.shape[0] > 32:
                st_s = max(1, sub_pts.shape[0] // 32)
                sub_pts = sub_pts[::st_s]
                sub_labs = sub_labs[::st_s]
            c_vec = np.array([cx, cy, cz], dtype=np.float64)
            proj_cam = (sub_pts - c_vec[None, :]) @ cam_toward_vec
            max_span = max(1.0, float(np.max(np.abs(proj_cam))))
            norm_p = proj_cam / max_span
            sample_w = np.where(
                proj_cam >= 0.0,
                w_front_cfg * (1.0 + norm_p),
                w_rear_cfg * np.maximum(0.1, 1.0 + 0.5 * norm_p),
            )
            sample_w = sample_w / max(1.0e-9, float(np.sum(sample_w)))
            # Evaluate weighted expected perceptual error across samples (`(S, P)` broadcast)
            s_cnt = sub_labs.shape[0]
            p_cnt = cand_labs.shape[0]
            s_rep = np.repeat(sub_labs, p_cnt, axis=0)
            c_tile = np.tile(cand_labs, (s_cnt, 1))
            de_mat = ciede2000_delta_e(s_rep, c_tile).reshape(s_cnt, p_cnt)
            raw_de = sample_w @ de_mat

        scored_de = raw_de + finish_penalties
        best_idx = int(np.argmin(scored_de))
        state.set_part_color(iid, int(cand_codes[best_idx]))
        delta_es.append(float(raw_de[best_idx]))

    return float(np.mean(delta_es)) if delta_es else 0.0


def count_exposed_upward_studs(state: AssemblyState) -> int:
    """Count unmatched upward (`+Z`) male stud ports across the entire `AssemblyState`."""
    matched = state.matched_port_pairs
    cnt = 0
    for iid, pl in state.placed_parts.items():
        prim = get_brick_primitive(pl.part_id)
        for p in prim.ports:
            if p.polarity == PortPolarity.STUD and p.connector_type in (
                ConnectorType.STANDARD_STUD,
                ConnectorType.HOLLOW_STUD,
            ):
                if (iid, p.port_id) in matched:
                    continue
                wf = pl.pose.compose(p.local_frame)
                if wf.z_axis == (0, 0, 1):
                    cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# Fast Cached 2-LDU CSG Pair Clearance Helper (Guarantees 100% Independent Replay)
# ---------------------------------------------------------------------------

_PACK_CSG_BIAS: int = 524288


@lru_cache(maxsize=4096)
def _cached_local_csg_2ldu(part_id: str, rot_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Origin-centered `(coords_M3_int64, bounds_2x3_int64)` at `2 LDU` (`0.8 mm`) resolution."""
    prim = get_brick_primitive(part_id)
    rot_mat = np.asarray(rot_id_to_matrix(rot_id), dtype=np.float64)

    def _ras_box(lb: LDUBox, shrink: float) -> np.ndarray:
        bx0, by0, bz0 = float(lb.xmin), float(lb.ymin), float(lb.zmin)
        bx1, by1, bz1 = float(lb.xmax), float(lb.ymax), float(lb.zmax)
        if (bx1 - bx0) <= 2.0 * shrink or (by1 - by0) <= 2.0 * shrink or (bz1 - bz0) <= 2.0 * shrink:
            return np.empty((0, 3), dtype=np.int64)
        corners = np.array(
            [[x, y, z] for x in (bx0, bx1) for y in (by0, by1) for z in (bz0, bz1)],
            dtype=np.float64,
        )
        cw = corners @ rot_mat.T
        ix0, ix1 = int(math.floor(float(np.min(cw[:, 0])) / 2.0)), int(math.ceil(float(np.max(cw[:, 0])) / 2.0))
        iy0, iy1 = int(math.floor(float(np.min(cw[:, 1])) / 2.0)), int(math.ceil(float(np.max(cw[:, 1])) / 2.0))
        iz0, iz1 = int(math.floor(float(np.min(cw[:, 2])) / 2.0)), int(math.ceil(float(np.max(cw[:, 2])) / 2.0))
        gx = (np.arange(ix0, ix1, dtype=np.float64) + 0.5) * 2.0
        gy = (np.arange(iy0, iy1, dtype=np.float64) + 0.5) * 2.0
        gz = (np.arange(iz0, iz1, dtype=np.float64) + 0.5) * 2.0
        if gx.size == 0 or gy.size == 0 or gz.size == 0:
            return np.empty((0, 3), dtype=np.int64)
        xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
        pts_l = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1) @ rot_mat
        m = (
            (pts_l[:, 0] >= bx0 + shrink)
            & (pts_l[:, 0] <= bx1 - shrink)
            & (pts_l[:, 1] >= by0 + shrink)
            & (pts_l[:, 1] <= by1 - shrink)
            & (pts_l[:, 2] >= bz0 + shrink)
            & (pts_l[:, 2] <= bz1 - shrink)
        )
        if not np.any(m):
            return np.empty((0, 3), dtype=np.int64)
        ii, jj, kk = np.meshgrid(
            np.arange(ix0, ix1, dtype=np.int64),
            np.arange(iy0, iy1, dtype=np.int64),
            np.arange(iz0, iz1, dtype=np.int64),
            indexing="ij",
        )
        return np.stack([ii.ravel()[m], jj.ravel()[m], kk.ravel()[m]], axis=1)

    def _pack(c: np.ndarray) -> np.ndarray:
        if c.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)
        return np.unique(
            ((c[:, 0] + _PACK_CSG_BIAS) << 40)
            | ((c[:, 1] + _PACK_CSG_BIAS) << 20)
            | (c[:, 2] + _PACK_CSG_BIAS)
        )

    chunks = [_ras_box(hb, 0.20) for hb in prim.hull_boxes]
    pos_c = (
        np.concatenate([c for c in chunks if c.shape[0] > 0], axis=0)
        if any(c.shape[0] > 0 for c in chunks)
        else np.empty((0, 3), dtype=np.int64)
    )
    v_keys = _pack(pos_c)

    for p in prim.ports:
        if p.polarity == PortPolarity.ANTI_STUD:
            pk = get_admissible_socket_pocket_ldu(p.connector_type)
            wf = LDUFrame(rot_id, (0, 0, 0)).compose(p.local_frame)
            rm_p = np.asarray(rot_id_to_matrix(wf.rot_id), dtype=np.float64)
            tv_p = np.asarray(wf.pos, dtype=np.float64)
            # Transform pocket by wf
            bx0, by0, bz0 = float(pk.xmin - 1), float(pk.ymin - 1), float(pk.zmin)
            bx1, by1, bz1 = float(pk.xmax + 1), float(pk.ymax + 1), float(pk.zmax)
            corners = np.array(
                [[x, y, z] for x in (bx0, bx1) for y in (by0, by1) for z in (bz0, bz1)],
                dtype=np.float64,
            )
            cw = (corners @ rm_p.T) + tv_p[None, :]
            ix0, ix1 = int(math.floor(float(np.min(cw[:, 0])) / 2.0)), int(math.ceil(float(np.max(cw[:, 0])) / 2.0))
            iy0, iy1 = int(math.floor(float(np.min(cw[:, 1])) / 2.0)), int(math.ceil(float(np.max(cw[:, 1])) / 2.0))
            iz0, iz1 = int(math.floor(float(np.min(cw[:, 2])) / 2.0)), int(math.ceil(float(np.max(cw[:, 2])) / 2.0))
            gx = (np.arange(ix0, ix1, dtype=np.float64) + 0.5) * 2.0
            gy = (np.arange(iy0, iy1, dtype=np.float64) + 0.5) * 2.0
            gz = (np.arange(iz0, iz1, dtype=np.float64) + 0.5) * 2.0
            if gx.size > 0 and gy.size > 0 and gz.size > 0:
                xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
                pts_l = (np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1) - tv_p[None, :]) @ rm_p
                m = (
                    (pts_l[:, 0] >= bx0)
                    & (pts_l[:, 0] <= bx1)
                    & (pts_l[:, 1] >= by0)
                    & (pts_l[:, 1] <= by1)
                    & (pts_l[:, 2] >= bz0)
                    & (pts_l[:, 2] <= bz1)
                )
                if np.any(m):
                    ii, jj, kk = np.meshgrid(
                        np.arange(ix0, ix1, dtype=np.int64),
                        np.arange(iy0, iy1, dtype=np.int64),
                        np.arange(iz0, iz1, dtype=np.int64),
                        indexing="ij",
                    )
                    ck = _pack(np.stack([ii.ravel()[m], jj.ravel()[m], kk.ravel()[m]], axis=1))
                    v_keys = np.setdiff1d(v_keys, ck, assume_unique=True)

    for p in prim.ports:
        if p.local_protrusion_box is not None:
            pb = p.local_protrusion_box
            wf = LDUFrame(rot_id, (0, 0, 0)).compose(p.local_frame)
            rm_p = np.asarray(rot_id_to_matrix(wf.rot_id), dtype=np.float64)
            tv_p = np.asarray(wf.pos, dtype=np.float64)
            bx0, by0, bz0 = float(pb.xmin), float(pb.ymin), float(pb.zmin)
            bx1, by1, bz1 = float(pb.xmax), float(pb.ymax), float(pb.zmax)
            corners = np.array(
                [[x, y, z] for x in (bx0, bx1) for y in (by0, by1) for z in (bz0, bz1)],
                dtype=np.float64,
            )
            cw = (corners @ rm_p.T) + tv_p[None, :]
            ix0, ix1 = int(math.floor(float(np.min(cw[:, 0])) / 2.0)), int(math.ceil(float(np.max(cw[:, 0])) / 2.0))
            iy0, iy1 = int(math.floor(float(np.min(cw[:, 1])) / 2.0)), int(math.ceil(float(np.max(cw[:, 1])) / 2.0))
            iz0, iz1 = int(math.floor(float(np.min(cw[:, 2])) / 2.0)), int(math.ceil(float(np.max(cw[:, 2])) / 2.0))
            gx = (np.arange(ix0, ix1, dtype=np.float64) + 0.5) * 2.0
            gy = (np.arange(iy0, iy1, dtype=np.float64) + 0.5) * 2.0
            gz = (np.arange(iz0, iz1, dtype=np.float64) + 0.5) * 2.0
            if gx.size > 0 and gy.size > 0 and gz.size > 0:
                xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
                pts_l = (np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1) - tv_p[None, :]) @ rm_p
                m = (
                    (pts_l[:, 0] >= bx0 + 0.25)
                    & (pts_l[:, 0] <= bx1 - 0.25)
                    & (pts_l[:, 1] >= by0 + 0.25)
                    & (pts_l[:, 1] <= by1 - 0.25)
                    & (pts_l[:, 2] >= bz0 + 0.25)
                    & (pts_l[:, 2] <= bz1 - 0.25)
                )
                if np.any(m):
                    ii, jj, kk = np.meshgrid(
                        np.arange(ix0, ix1, dtype=np.int64),
                        np.arange(iy0, iy1, dtype=np.int64),
                        np.arange(iz0, iz1, dtype=np.int64),
                        indexing="ij",
                    )
                    pk_k = _pack(np.stack([ii.ravel()[m], jj.ravel()[m], kk.ravel()[m]], axis=1))
                    v_keys = np.union1d(v_keys, pk_k)

    if v_keys.size == 0:
        return np.empty((0,), dtype=np.int64), np.zeros((2, 3), dtype=np.int64)
    ix = (v_keys >> 40) - _PACK_CSG_BIAS
    iy = ((v_keys >> 20) & 0xFFFFF) - _PACK_CSG_BIAS
    iz = (v_keys & 0xFFFFF) - _PACK_CSG_BIAS
    bounds = np.array(
        [
            [int(np.min(ix)), int(np.min(iy)), int(np.min(iz))],
            [int(np.max(ix)), int(np.max(iy)), int(np.max(iz))],
        ],
        dtype=np.int64,
    )
    return v_keys, bounds


def _world_csg_keys_even(part_id: str, pose: LDUFrame) -> Tuple[np.ndarray, np.ndarray]:
    v_keys, bounds = _cached_local_csg_2ldu(part_id, pose.rot_id)
    if v_keys.size == 0:
        return v_keys, bounds
    sx = int(pose.pos[0] // 2)
    sy = int(pose.pos[1] // 2)
    sz = int(pose.pos[2] // 2)
    delta_key = np.int64(sx * (1 << 40) + sy * (1 << 20) + sz)
    sh = np.array([sx, sy, sz], dtype=np.int64)
    return v_keys + delta_key, bounds + sh[None, :]


@lru_cache(maxsize=8192)
def _fast_rel_mask6(
    part_id: str,
    rot_id: int,
    rx_mod: int,
    ry_mod: int,
    rz_mod: int,
) -> Tuple[int, int, int, int, int, int, int, bool, np.ndarray]:
    """
    Translation-invariant relative `5x5x6 LDU` sub-voxel mask for `(part_id, rot_id)` at sub-grid
    remainder `(rx_mod = wx % 5, ry_mod = wy % 5, rz_mod = wz % 6)`.
    Returns `(rx0, rx1, ry0, ry1, rz0, rz1, min_z_local_ldu, is_full_cuboid, rel_mask_3d)`.
    """
    prim = get_brick_primitive(part_id)
    rf = LDUFrame(rot_id, (0, 0, 0))
    rboxes = [b.transform(rf) for b in prim.hull_boxes]
    if not rboxes:
        return (0, 0, 0, 0, 0, 0, 0, True, np.empty((0, 0, 0), dtype=bool))
    min_z_loc = min(b.zmin for b in rboxes)
    rx0 = min(int(math.floor((b.xmin + rx_mod) / 5.0)) for b in rboxes)
    rx1 = max(int(math.ceil((b.xmax + rx_mod) / 5.0)) for b in rboxes)
    ry0 = min(int(math.floor((b.ymin + ry_mod) / 5.0)) for b in rboxes)
    ry1 = max(int(math.ceil((b.ymax + ry_mod) / 5.0)) for b in rboxes)
    rz0 = min(int(math.floor((b.zmin + rz_mod) / 6.0)) for b in rboxes)
    rz1 = max(int(math.ceil((b.zmax + rz_mod) / 6.0)) for b in rboxes)
    dx, dy, dz = rx1 - rx0, ry1 - ry0, rz1 - rz0
    if dx <= 0 or dy <= 0 or dz <= 0:
        return (rx0, rx1, ry0, ry1, rz0, rz1, min_z_loc, True, np.empty((0, 0, 0), dtype=bool))
    if len(rboxes) == 1:
        return (rx0, rx1, ry0, ry1, rz0, rz1, min_z_loc, True, np.empty((0, 0, 0), dtype=bool))
    loc = np.zeros((dx, dy, dz), dtype=bool)
    for b in rboxes:
        bx0 = int(math.floor((b.xmin + rx_mod) / 5.0)) - rx0
        bx1 = int(math.ceil((b.xmax + rx_mod) / 5.0)) - rx0
        by0 = int(math.floor((b.ymin + ry_mod) / 5.0)) - ry0
        by1 = int(math.ceil((b.ymax + ry_mod) / 5.0)) - ry0
        bz0 = int(math.floor((b.zmin + rz_mod) / 6.0)) - rz0
        bz1 = int(math.ceil((b.zmax + rz_mod) / 6.0)) - rz0
        if bx1 > bx0 and by1 > by0 and bz1 > bz0:
            loc[bx0:bx1, by0:by1, bz0:bz1] = True
    if bool(np.all(loc)):
        return (rx0, rx1, ry0, ry1, rz0, rz1, min_z_loc, True, np.empty((0, 0, 0), dtype=bool))
    return (rx0, rx1, ry0, ry1, rz0, rz1, min_z_loc, False, loc)


@lru_cache(maxsize=2048)
def _fast_upward_stud_templates(
    part_id: str,
) -> Tuple[Tuple[int, int, int, int, Tuple[Tuple[int, int, int, int, int, int], ...]], ...]:
    """
    Fast `C4_Z_ROT_IDS`-only upward stud mating templates `(rid, dx, dy, dz, rel_boxes_from_stud)`
    with rotational symmetry deduplication (`~20x` faster than 24-rotation `_get_anchor_templates`).
    """
    prim = get_brick_primitive(part_id)
    anti_z = [
        p
        for p in prim.ports
        if p.polarity == PortPolarity.ANTI_STUD
        and p.local_frame.z_axis == (0, 0, 1)
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    ]
    if not anti_z or not prim.hull_boxes:
        return ()
    out: List[Tuple[int, int, int, int, Tuple[Tuple[int, int, int, int, int, int], ...]]] = []
    seen_geom: Set[Tuple[Tuple[int, int, int, int, int, int], ...]] = set()
    for rid in C4_Z_ROT_IDS:
        rf = LDUFrame(rid, (0, 0, 0))
        tboxes = [hb.transform(rf) for hb in prim.hull_boxes]
        for p in anti_z:
            ax, ay, az = rf.compose(p.local_frame).pos
            dx, dy, dz = -ax, -ay, -az
            rboxes = tuple(
                sorted(
                    (
                        tb.xmin + dx,
                        tb.ymin + dy,
                        tb.zmin + dz,
                        tb.xmax + dx,
                        tb.ymax + dy,
                        tb.zmax + dz,
                    )
                    for tb in tboxes
                )
            )
            if rboxes in seen_geom:
                continue
            seen_geom.add(rboxes)
            out.append((int(rid), dx, dy, dz, rboxes))
    return tuple(out)


@lru_cache(maxsize=1024)
def _fast_slope_grid_templates(
    part_id: str,
) -> Tuple[
    Tuple[
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        Tuple[Tuple[int, int, int, int, int, int], ...],
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int], ...],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ],
    ...,
]:
    """
    Precomputed grid-aligned `C4_Z_ROT_IDS` slope/wedge/curve/panel templates:
    `(rid, dx, dy, dz, drx0, drx1, dry0, dry1, h_pl, rboxes, rel_top_studs, rel_banti, rel_stud_clash, masks_by_zmod3)`.
    Eliminates per-cell box bounds, port transforms, and sub-voxel mask rasterization at runtime.
    """
    prim = get_brick_primitive(part_id)
    out = []
    for rid, dx, dy, dz, rboxes in _fast_upward_stud_templates(part_id):
        rel_zmin = min(b[2] for b in rboxes)
        rel_zmax = max(b[5] for b in rboxes)
        h_ldu = rel_zmax - rel_zmin
        if rel_zmin != 0 or h_ldu not in (8, 16, 24, 32, 48, 72):
            continue
        rel_xmin = min(b[0] for b in rboxes)
        rel_xmax = max(b[3] for b in rboxes)
        rel_ymin = min(b[1] for b in rboxes)
        rel_ymax = max(b[4] for b in rboxes)
        drx0 = int(math.floor((10 + rel_xmin) / 20.0))
        drx1 = int(math.ceil((10 + rel_xmax) / 20.0))
        dry0 = int(math.floor((10 + rel_ymin) / 20.0))
        dry1 = int(math.ceil((10 + rel_ymax) / 20.0))
        if drx1 <= drx0 or dry1 <= dry0:
            continue
        cell_xmin = drx0 * 20 - 10
        cell_ymin = dry0 * 20 - 10
        h_pl = h_ldu // 8
        lf_base = LDUFrame(rid, (dx, dy, dz))
        tstuds: Set[Tuple[int, int]] = set()
        banti: Set[Tuple[int, int]] = set()
        for pt_sm in prim.ports:
            if pt_sm.polarity == PortPolarity.STUD:
                wf_loc = lf_base.compose(pt_sm.local_frame)
                if wf_loc.z_axis == (0, 0, 1) and wf_loc.pos[2] == rel_zmax:
                    tstuds.add(((10 + wf_loc.pos[0]) // 20, (10 + wf_loc.pos[1]) // 20))
            elif pt_sm.polarity == PortPolarity.ANTI_STUD:
                wf_loc = lf_base.compose(pt_sm.local_frame)
                if wf_loc.z_axis == (0, 0, 1) and wf_loc.pos[2] == rel_zmin:
                    banti.add(((10 + wf_loc.pos[0]) // 20, (10 + wf_loc.pos[1]) // 20))
        area_cells = (drx1 - drx0) * (dry1 - dry0)
        min_banti_req = max(1, (area_cells + 1) // 2) if h_ldu <= 24 else max(1, area_cells - 1)
        if len(banti) < min_banti_req:
            continue
        stud_clash: Set[Tuple[int, int]] = set()
        for cx_s in range(drx0, drx1):
            for cy_s in range(dry0, dry1):
                if (cx_s, cy_s) in banti:
                    continue
                sx0_c, sx1_c = cx_s * 20 - 4, cx_s * 20 + 4
                sy0_c, sy1_c = cy_s * 20 - 4, cy_s * 20 + 4
                if any(
                    b[0] < sx1_c and b[3] > sx0_c and b[1] < sy1_c and b[4] > sy0_c and b[2] < 4 and b[5] > 0
                    for b in rboxes
                ):
                    stud_clash.add((cx_s, cy_s))
        masks_list: List[np.ndarray] = []
        nx6 = 4 * (drx1 - drx0)
        ny6 = 4 * (dry1 - dry0)
        for r_mod in (0, 1, 2):
            z0_ref = 8 * r_mod
            sz0_r = z0_ref // 6
            sz1_r = int(math.ceil((z0_ref + h_ldu) / 6.0))
            nz6 = sz1_r - sz0_r
            m_arr = np.zeros((nx6, ny6, nz6), dtype=bool)
            for b in rboxes:
                kx0 = max(0, int(math.floor((b[0] - cell_xmin) / 5.0)))
                kx1 = min(nx6, int(math.ceil((b[3] - cell_xmin) / 5.0)))
                ky0 = max(0, int(math.floor((b[1] - cell_ymin) / 5.0)))
                ky1 = min(ny6, int(math.ceil((b[4] - cell_ymin) / 5.0)))
                kz0 = max(0, int(math.floor((b[2] + z0_ref) / 6.0)) - sz0_r)
                kz1 = min(nz6, int(math.ceil((b[5] + z0_ref) / 6.0)) - sz0_r)
                if kx1 > kx0 and ky1 > ky0 and kz1 > kz0:
                    m_arr[kx0:kx1, ky0:ky1, kz0:kz1] = True
            masks_list.append(m_arr)
        rel_occ_cells = tuple(
            (drx0 + dx_m, dry0 + dy_m)
            for dx_m in range(drx1 - drx0)
            for dy_m in range(dry1 - dry0)
            if bool(masks_list[0][dx_m * 4 : (dx_m + 1) * 4, dy_m * 4 : (dy_m + 1) * 4, :].any())
        )
        out.append(
            (
                rid,
                dx,
                dy,
                dz,
                drx0,
                drx1,
                dry0,
                dry1,
                h_pl,
                rboxes,
                tuple(sorted(tstuds)),
                tuple(sorted(banti)),
                tuple(sorted(stud_clash)),
                rel_occ_cells,
                (masks_list[0], masks_list[1], masks_list[2]),
            )
        )
    return tuple(out)


def _box_hits_forbidden(box: LDUBox, forbidden: Sequence[LDUBox]) -> bool:
    return any(box.intersects(fb) for fb in forbidden)


def _rect_templates(
    allowed_ids: Sequence[str],
    height_ldu: int,
    studless_tiles: bool = False,
) -> List[Tuple[str, int, int]]:
    """Return `(part_id, studs_x, studs_y)` rectangular templates sorted largest-area-first."""
    out: List[Tuple[str, int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for pid in allowed_ids:
        prim = get_brick_primitive(pid)
        if prim.is_snot or len(prim.hull_boxes) != 1:
            continue
        b = prim.hull_boxes[0]
        if (b.zmax - b.zmin) != height_ldu:
            continue
        sx = (b.xmax - b.xmin) // 20
        sy = (b.ymax - b.ymin) // 20
        if sx <= 0 or sy <= 0 or (b.xmax - b.xmin) != sx * 20 or (b.ymax - b.ymin) != sy * 20:
            continue
        req_studs = 0 if studless_tiles else sx * sy
        anti_ok = (
            len(prim.anti_stud_ports) >= sx * sy
            if studless_tiles
            else len(prim.anti_stud_ports) == sx * sy
        )
        if len(prim.stud_ports) != req_studs or not anti_ok:
            continue
        key = (min(sx, sy), max(sx, sy))
        if key not in seen:
            seen.add(key)
            out.append((pid, sx, sy))
    out.sort(key=lambda t: (-(t[1] * t[2]), -max(t[1], t[2]), t[0]))
    return out


def _tile_boolean_layer_with_rects(
    occ_xy: np.ndarray,
    z0: int,
    ox: int,
    oy: int,
    templates: Sequence[Tuple[str, int, int]],
    prefer_x_major: bool,
    parity_shift: int,
    rem_inv: Optional[Dict[str, int]],
    supported_below_xy: Optional[np.ndarray] = None,
    shell_running_bond: bool = False,
) -> List[Tuple[str, LDUFrame]]:
    """
    Tile a 2D `(nx, ny)` boolean mask at height `z0` and sub-stud shift `(ox, oy)` using
    available rectangular brick or plate templates:
      - Pass A: Span unsupported arch/cantilever cells (`~supported_below_xy`) first, strongly
        preferring double-ended pier-to-pier lintel bridges so zero bricks float in midair.
      - Pass B: Tile remaining supported cells with seam-staggered orientation preference.
    """
    # Check `1x1` scarcity (`Phase 11 Ext 2.2`): snap odd-length 1D runs to even parity when `1x1` stock is zero/scarce
    if rem_inv is not None:
        s_1x1 = int(rem_inv.get("3005", 0)) + int(rem_inv.get("3024", 0)) // 3
        has_any_1x1_tmpl = any(t[1] == 1 and t[2] == 1 and rem_inv.get(t[0], 0) > 0 for t in templates)
        if not has_any_1x1_tmpl or s_1x1 <= 2:
            from topo_brick.kernel.colored_inventory_allocator import (
                apply_even_parity_morphological_snap,
            )

            occ_xy = apply_even_parity_morphological_snap(occ_xy, supported_below_xy)

    nx, ny = occ_xy.shape
    covered = np.zeros((nx, ny), dtype=bool)
    placements: List[Tuple[str, LDUFrame]] = []

    # Precompute `(pid, rid, wx, wy, b_xmin, b_ymin, b_zmin)` per template
    tmpl_orients: List[Tuple[str, List[Tuple[int, int, int, int, int, int]]]] = []
    for pid, _, _ in templates:
        ors: List[Tuple[int, int, int, int, int, int]] = []
        seen_wh: Set[Tuple[int, int]] = set()
        for rid in (0, 9):
            b_loc = get_brick_primitive(pid).hull_boxes[0].transform(LDUFrame(rid, (0, 0, 0)))
            wx = (b_loc.xmax - b_loc.xmin) // 20
            wy = (b_loc.ymax - b_loc.ymin) // 20
            if (wx, wy) not in seen_wh:
                seen_wh.add((wx, wy))
                ors.append((rid, wx, wy, b_loc.xmin, b_loc.ymin, b_loc.zmin))
        if prefer_x_major:
            ors.sort(key=lambda o: (-o[1], -o[2]))
        else:
            ors.sort(key=lambda o: (-o[2], -o[1]))
        tmpl_orients.append((pid, ors))

    # Pass A: Bridge unsupported arch/cantilever cells first (farthest from piers first so mid-span cells never get stranded!)
    if supported_below_xy is not None:
        sup_pts_arr = np.argwhere(supported_below_xy)
        while True:
            unsup_cells = np.argwhere(occ_xy & (~covered) & (~supported_below_xy))
            if unsup_cells.shape[0] == 0:
                break
            if sup_pts_arr.shape[0] > 0:
                dists_p = np.min(
                    np.abs(unsup_cells[:, None, 0] - sup_pts_arr[None, :, 0])
                    + np.abs(unsup_cells[:, None, 1] - sup_pts_arr[None, :, 1]),
                    axis=1,
                )
                pick_i = int(np.argmax(dists_p))
                ux0, uy0 = int(unsup_cells[pick_i, 0]), int(unsup_cells[pick_i, 1])
            else:
                ux0, uy0 = int(unsup_cells[0, 0]), int(unsup_cells[0, 1])
            best_sc = -1e9
            best_cand: Optional[Tuple[str, int, int, int, int, int, int, int, int]] = None
            for pid, ors in tmpl_orients:
                if rem_inv is not None and rem_inv.get(pid, 0) <= 0:
                    continue
                for rid, wx, wy, bxmin, bymin, bzmin in ors:
                    for bx in range(max(0, ux0 - wx + 1), min(ux0 + 1, nx - wx + 1)):
                        for by in range(max(0, uy0 - wy + 1), min(uy0 + 1, ny - wy + 1)):
                            if not np.all(occ_xy[bx : bx + wx, by : by + wy]):
                                continue
                            if np.any(covered[bx : bx + wx, by : by + wy]):
                                continue
                            sup_sl = supported_below_xy[bx : bx + wx, by : by + wy]
                            n_sup = int(np.count_nonzero(sup_sl))
                            if n_sup == 0:
                                continue
                            n_unsup = int(wx * wy - n_sup)
                            # Check double-ended arch bridge along X or Y
                            two_end_x = wx >= 3 and bool(np.any(sup_sl[0, :])) and bool(np.any(sup_sl[-1, :]))
                            two_end_y = wy >= 3 and bool(np.any(sup_sl[:, 0])) and bool(np.any(sup_sl[:, -1]))
                            # Check balanced CoM or multi-stud pier anchor (`>= 2` pier studs counterbalancing `<= 4`-stud cantilevers)
                            bal_bonus = (
                                220.0
                                if (two_end_x or two_end_y)
                                else (75.0 if (n_sup >= n_unsup or n_sup >= 2) else 0.0)
                            )
                            sc = bal_bonus + 28.0 * float(n_unsup) + 12.0 * float(n_sup) + float(wx * wy)
                            if sc > best_sc:
                                best_sc = sc
                                best_cand = (pid, rid, bx, by, wx, wy, bxmin, bymin, bzmin)
            if best_cand is None:
                # Mark `(ux0, uy0)` skipped if unreachable by any single span from piers
                covered[ux0, uy0] = True
                continue
            pid_c, rid_c, bx_c, by_c, wx_c, wy_c, bxmin_c, bymin_c, bzmin_c = best_cand
            tx = bx_c * 20 + ox - bxmin_c
            ty = by_c * 20 + oy - bymin_c
            tz = z0 - bzmin_c
            placements.append((pid_c, LDUFrame(rid_c, (tx, ty, tz))))
            covered[bx_c : bx_c + wx_c, by_c : by_c + wy_c] = True
            if rem_inv is not None:
                rem_inv[pid_c] = rem_inv.get(pid_c, 0) - 1

    # Pass B: Seam-staggered English/Flemish tiling of all remaining supported cells
    pass_b_start_idx = len(placements)
    covered_before_b = covered.copy()
    rem_inv_before_b = dict(rem_inv) if rem_inv is not None else None

    xs_order = list(range(parity_shift, nx)) + list(range(0, parity_shift))
    ys_order = list(range(parity_shift, ny)) + list(range(0, parity_shift))
    stretcher_first_orients = sorted(
        tmpl_orients,
        key=lambda item: (
            0 if (min(item[1][0][1], item[1][0][2]) == 1 and max(item[1][0][1], item[1][0][2]) >= 3) else 1,
            -(item[1][0][1] * item[1][0][2]),
        ),
    )
    med_targets = ((2, 2), (2, 3), (1, 4), (1, 3))
    z_step_idx = (z0 // 8)
    used_med_lead = 0

    occ_l = occ_xy.tolist()
    cov_l = covered.tolist()
    sup_l = supported_below_xy.tolist() if supported_below_xy is not None else None

    if shell_running_bond:
        # Safeguard #3 (`Phase 11 Ext 3`): On hollow-shell courses, alternate scan axis & orientation
        # parity with zero corner wrap shift so X-tangent spans cross all corners/Y-ribs on even courses
        # and Y-tangent spans cross all corners/X-ribs on odd courses (zero vertical chimney seams!).
        scan_pairs = (
            [(ix_s, iy_s) for iy_s in range(ny) for ix_s in range(nx)]
            if prefer_x_major
            else [(ix_s, iy_s) for ix_s in range(nx) for iy_s in range(ny)]
        )
    else:
        scan_pairs = [(ix_s, iy_s) for ix_s in xs_order for iy_s in ys_order]

    for ix, iy in scan_pairs:
        if not occ_l[ix][iy] or cov_l[ix][iy]:
            continue
        if True:
            placed_here = False
            if (not shell_running_bond) and used_med_lead < 3 and ((z0 > 0 and (nx * ny) >= 4) or (z0 == 0 and (nx * ny) >= 12)):
                cur_t = med_targets[(nx * 3 + ny * 5 + z_step_idx + used_med_lead) % len(med_targets)]
                use_list = sorted(
                    tmpl_orients,
                    key=lambda item: (
                        0
                        if (
                            min(item[1][0][1], item[1][0][2]) == cur_t[0]
                            and max(item[1][0][1], item[1][0][2]) == cur_t[1]
                        )
                        else 1,
                        -(item[1][0][1] * item[1][0][2]),
                    ),
                )
                used_med_lead += 1
            elif shell_running_bond:
                use_list = tmpl_orients
            elif parity_shift == 1 and (ix == 0 or iy == 0 or ix == nx - 1 or iy == ny - 1):
                use_list = stretcher_first_orients
            else:
                use_list = tmpl_orients
            for pid, ors in use_list:
                if rem_inv is not None and rem_inv.get(pid, 0) <= 0:
                    continue
                for rid, wx, wy, bxmin, bymin, bzmin in ors:
                    if ix + wx > nx or iy + wy > ny:
                        continue
                    rect_ok = True
                    has_sup_rect = sup_l is None
                    for kx in range(ix, ix + wx):
                        o_kx = occ_l[kx]
                        c_kx = cov_l[kx]
                        s_kx = sup_l[kx] if sup_l is not None else None
                        for ky in range(iy, iy + wy):
                            if not o_kx[ky] or c_kx[ky]:
                                rect_ok = False
                                break
                            if not has_sup_rect and s_kx[ky]:  # type: ignore[index]
                                has_sup_rect = True
                        if not rect_ok:
                            break
                    if not rect_ok or not has_sup_rect:
                        continue
                    tx = ix * 20 + ox - bxmin
                    ty = iy * 20 + oy - bymin
                    tz = z0 - bzmin
                    placements.append((pid, LDUFrame(rid, (tx, ty, tz))))
                    for kx in range(ix, ix + wx):
                        c_kx = cov_l[kx]
                        for ky in range(iy, iy + wy):
                            c_kx[ky] = True
                    if rem_inv is not None:
                        rem_inv[pid] = rem_inv.get(pid, 0) - 1
                    placed_here = True
                    break
                if placed_here:
                    break

    # Extension 2.2: If finite `rem_inv` left supported cells uncovered under greedy slicing,
    # rewind Pass B on this course and solve exact bounded-inventory set-packing MILP (`< 3 ms`)!
    if rem_inv is not None and rem_inv_before_b is not None:
        cov_after_b = np.asarray(cov_l, dtype=bool)
        sup_mask_chk = supported_below_xy if supported_below_xy is not None else np.ones_like(occ_xy)
        uncov_sup = int(np.count_nonzero(occ_xy & (~cov_after_b) & sup_mask_chk))
        if uncov_sup > 0:
            from topo_brick.kernel.colored_inventory_allocator import (
                solve_sparse_kit_course_set_packing,
            )

            cov_milp = covered_before_b.copy()
            inv_milp = dict(rem_inv_before_b)
            milp_pls = solve_sparse_kit_course_set_packing(
                occ_xy=occ_xy,
                covered_xy=cov_milp,
                z0=z0,
                ox=ox,
                oy=oy,
                templates=templates,
                prefer_x_major=prefer_x_major,
                parity_shift=parity_shift,
                rem_inv=inv_milp,
                supported_below_xy=supported_below_xy,
            )
            uncov_milp = int(np.count_nonzero(occ_xy & (~cov_milp) & sup_mask_chk))
            if uncov_milp <= uncov_sup and len(milp_pls) > 0:
                placements = placements[:pass_b_start_idx] + milp_pls
                rem_inv.clear()
                rem_inv.update(inv_milp)

    if shell_running_bond and pass_b_start_idx > 0:
        # Place 100%-supported wall/rib counterweight bricks on this course BEFORE overhanging corbel/ceiling spans!
        return placements[pass_b_start_idx:] + placements[:pass_b_start_idx]
    return placements




def _build_fine_target_field_from_occ6(occ6: np.ndarray) -> SparseTiledTargetField:
    """Pack `occ6` (`5x5x6 LDU` boolean grid) into XYZ-merged `LDUBox` cuboids (`~10x` fewer boxes)."""
    gx, gy, gz = occ6.shape
    z_rects: Dict[Tuple[int, int, int, int], List[Tuple[int, int]]] = {}
    for iz in range(gz):
        sl = occ6[:, :, iz]
        if not np.any(sl):
            continue
        z0, z1 = iz * 6, (iz + 1) * 6
        active_y: Dict[Tuple[int, int], Tuple[int, int]] = {}
        xy_list: List[Tuple[int, int, int, int]] = []
        for iy in range(gy):
            row = sl[:, iy]
            y0, y1 = iy * 5, (iy + 1) * 5
            runs_x: List[Tuple[int, int]] = []
            if np.any(row):
                ix = 0
                while ix < gx:
                    if not row[ix]:
                        ix += 1
                        continue
                    ix_end = ix + 1
                    while ix_end < gx and row[ix_end]:
                        ix_end += 1
                    runs_x.append((ix * 5, ix_end * 5))
                    ix = ix_end
            next_y: Dict[Tuple[int, int], Tuple[int, int]] = {}
            run_set = set(runs_x)
            for rx_k, (ry0_prev, ry1_prev) in active_y.items():
                if rx_k in run_set:
                    next_y[rx_k] = (ry0_prev, y1)
                else:
                    xy_list.append((rx_k[0], ry0_prev, rx_k[1], ry1_prev))
            for rx_k in runs_x:
                if rx_k not in next_y:
                    next_y[rx_k] = (y0, y1)
            active_y = next_y
        for rx_k, (ry0_prev, ry1_prev) in active_y.items():
            xy_list.append((rx_k[0], ry0_prev, rx_k[1], ry1_prev))
        for r_xy in xy_list:
            lst_z = z_rects.get(r_xy)
            if lst_z and lst_z[-1][1] == z0:
                lst_z[-1] = (lst_z[-1][0], z1)
            else:
                z_rects.setdefault(r_xy, []).append((z0, z1))
    boxes = [
        LDUBox(x0, y0, z0_b, x1, y1, z1_b)
        for (x0, y0, x1, y1), z_spans in z_rects.items()
        for z0_b, z1_b in z_spans
    ]
    return SparseTiledTargetField(boxes=tuple(boxes))



@lru_cache(maxsize=1)
def _get_ldconfig_full_palette_lab_and_penalties(
    preferred_finish: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from topo_brick.interface.ldraw_palette import LDRAW_FULL_PALETTE_RECORDS

    codes = np.array([int(r.code) for r in LDRAW_FULL_PALETTE_RECORDS], dtype=np.int32)
    srgb = np.array([r.rgb_u8 for r in LDRAW_FULL_PALETTE_RECORDS], dtype=np.float64)
    pref = str(preferred_finish).strip().upper() if preferred_finish is not None else None
    pens = np.zeros(len(LDRAW_FULL_PALETTE_RECORDS), dtype=np.float64)
    for i, r in enumerate(LDRAW_FULL_PALETTE_RECORDS):
        if int(r.code) in (16, 24):
            pens[i] = 50.0
            continue
        fin = str(r.finish).upper()
        if pref is not None:
            pens[i] = 0.0 if fin == pref else 14.0
        elif fin == "SOLID" and int(r.alpha) >= 255:
            pens[i] = 0.0
        elif fin in ("PEARLESCENT", "METALLIC"):
            pens[i] = 6.0
        elif fin == "RUBBER":
            pens[i] = 8.0
        else:
            pens[i] = 14.0
    return codes, srgb_to_cielab(srgb), pens


def _get_ldconfig_full_palette_lab() -> Tuple[np.ndarray, np.ndarray]:
    codes, labs, _ = _get_ldconfig_full_palette_lab_and_penalties(preferred_finish=None)
    return codes, labs
