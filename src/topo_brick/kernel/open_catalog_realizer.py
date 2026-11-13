"""
Phase 10 (`P10-WS2`): Open-Catalog Online `Top-K` Retriever, Phase-Aware `8 LDU` + `10 LDU`
Jumper Core, Exact Global `ΔIoU` 24-Orientation Sculpting, Studless Tile Skinning, and
Perceptual CIE `L*a*b*` (`CIEDE2000`) LDraw Color Mapper.

Integrates cleanly into `TopoBrickKernel.realize(...)`:
  - `part_retrieval="open_catalog_top_k"` (`retrieval_top_k=32`, `license_filter="all"|"cc_by_4_0_only"`)
  - `surface_refinement="open_catalog_sculpted"`
  - `semantic_style_hint=None | "geometric_optimal" | "smooth_studless"`
  - `vertex_colors_rgb: Optional[np.ndarray]` (`(V, 3)` float `[0, 1]` or uint8 `[0, 255]`)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import heapq
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    ConnectorType,
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.catalog_retrieval_index import (
    CATALOG_RETRIEVAL_INDEX,
    SCULPTURAL_TAXONOMY_CLASSES,
    CatalogRetrievalResult,
)
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame, rot_id_to_matrix
from topo_brick.interface.port_manifold import (
    AssemblyState,
    SpatialBoxHashIndex,
    _get_anchor_templates,
    get_admissible_socket_pocket_ldu,
)
from topo_brick.kernel.adaptive_surface_and_macro_planner import AdaptiveSurfaceTarget
from topo_brick.kernel.open_catalog_blind_portfolio import (
    _apply_terminal_tile_swap,
    refine_open_catalog_blind_portfolio,
    try_upfront_blind_sculpted_fastpath,
)
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
) -> float:
    """
    Assign official LDraw `color_id` to every placed part in `state` by sampling nearest mesh
    vertices `(verts_ldu, vertex_colors_rgb)` and matching `LDRAW_SOLID_PALETTE_LAB` via
    `CIEDE2000` (`ΔE_00`). Returns mean `ΔE_00` across all placed parts.
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
        if np.any(inside):
            target_lab = np.mean(v_lab[inside], axis=0)
        else:
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            cz = 0.5 * (zmin + zmax)
            d2 = (v_pts[:, 0] - cx) ** 2 + (v_pts[:, 1] - cy) ** 2 + (v_pts[:, 2] - cz) ** 2
            k_nn = min(8, v_pts.shape[0])
            nn_idx = np.argpartition(d2, k_nn - 1)[:k_nn]
            target_lab = np.mean(v_lab[nn_idx], axis=0)

        de_all = ciede2000_delta_e(target_lab[None, :], LDRAW_SOLID_PALETTE_LAB)
        best_idx = int(np.argmin(de_all))
        state.set_part_color(iid, int(_PALETTE_CODES[best_idx]))
        delta_es.append(float(de_all[best_idx]))

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
) -> List[Tuple[str, LDUFrame]]:
    """
    Tile a 2D `(nx, ny)` boolean mask at height `z0` and sub-stud shift `(ox, oy)` using
    available rectangular brick or plate templates:
      - Pass A: Span unsupported arch/cantilever cells (`~supported_below_xy`) first, strongly
        preferring double-ended pier-to-pier lintel bridges so zero bricks float in midair.
      - Pass B: Tile remaining supported cells with seam-staggered orientation preference.
    """
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

    for ix in xs_order:
        occ_ix = occ_l[ix]
        cov_ix = cov_l[ix]
        for iy in ys_order:
            if not occ_ix[iy] or cov_ix[iy]:
                continue
            placed_here = False
            if used_med_lead < 3 and ((z0 > 0 and (nx * ny) >= 4) or (z0 == 0 and (nx * ny) >= 12)):
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
    return placements


@dataclass(frozen=True, slots=True)
class OpenCatalogRealizeDiagnostics:
    """Telemetry produced by `realize_open_catalog_sculpted(...)`."""

    retrieval_result: CatalogRetrievalResult
    distinct_parts_placed: Tuple[str, ...]
    sculptural_parts_placed_count: int
    jumper_parts_placed_count: int
    snot_parts_placed_count: int
    smooth_tiles_placed_count: int
    exposed_upward_studs_count: int
    mean_color_delta_e00: float


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
    if not boxes:
        boxes.append(LDUBox(0, 0, 0, 20, 20, 8))
    return SparseTiledTargetField(boxes=tuple(boxes))


def realize_open_catalog_sculpted(
    kernel: Any,
    coarse_target_field: SparseTiledTargetField | Any,
    adaptive_target: AdaptiveSurfaceTarget,
    allowed_part_ids: Optional[Sequence[str]] = None,
    part_retrieval: str = "open_catalog_top_k",
    retrieval_top_k: int = 32,
    license_filter: str = "all",
    semantic_style_hint: Optional[str] = None,
    vertex_colors_rgb: Optional[np.ndarray] = None,
    verts_ldu: Optional[np.ndarray] = None,
    inventory_counts: Optional[Dict[str, int]] = None,
    support_policy: str = "minimal_permanent_pillars",
    foundation_policy: FoundationPolicy = FoundationPolicy.CLUTCHED_BASEPLATE,
    forbidden_support_boxes: Sequence[LDUBox] = (),
    _forced_plan: Optional[Tuple[List[Tuple[int, int]], Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]], int]] = None,
    _pre_ret: Optional[CatalogRetrievalResult] = None,
    _eval_only: bool = False,
    _max_waves: int = 2,
) -> Tuple[AssemblyState, SparseTiledTargetField, OpenCatalogRealizeDiagnostics]:
    """
    Synthesize a 6-DoF-certified open-catalog sculpted LEGO assembly directly from
    `adaptive_target` (`5x5x2` & `5x5x6 LDU` continuous mesh grids) + `CATALOG_RETRIEVAL_INDEX`.
    """
    from topo_brick.kernel.construction_kernel import check_monotone_insertion_corridor
    # 1. Pad `occ6` and `occ2` by +1 cell (`+4` sub-voxels in XY, `+4`/`+12` in Z) so boundary
    #    jumpers, wings, and SNOT skins never clip at `20*nx_studs`!
    nx_s = adaptive_target.nx_studs + 1
    ny_s = adaptive_target.ny_studs + 1
    nz_c = adaptive_target.nz_courses + 1
    n_plates_z = 3 * nz_c

    occ6 = np.zeros((4 * nx_s, 4 * ny_s, 4 * nz_c), dtype=bool)
    s6x, s6y, s6z = adaptive_target.occ_5x5x6.shape
    occ6[:s6x, :s6y, :s6z] = adaptive_target.occ_5x5x6

    occ2 = np.zeros((4 * nx_s, 4 * ny_s, 12 * nz_c), dtype=bool)
    s2x, s2y, s2z = adaptive_target.occ_5x5x2.shape
    occ2[:s2x, :s2y, :s2z] = adaptive_target.occ_5x5x2

    # Strictly mask out forbidden architectural openings (`100%` opening clearance invariant)
    for fb in forbidden_support_boxes:
        fx0 = max(0, int(math.floor(fb.xmin / 5.0)))
        fx1 = min(occ6.shape[0], int(math.ceil(fb.xmax / 5.0)))
        fy0 = max(0, int(math.floor(fb.ymin / 5.0)))
        fy1 = min(occ6.shape[1], int(math.ceil(fb.ymax / 5.0)))
        fz6_0 = max(0, int(math.floor(fb.zmin / 6.0)))
        fz6_1 = min(occ6.shape[2], int(math.ceil(fb.zmax / 6.0)))
        fz2_0 = max(0, int(math.floor(fb.zmin / 2.0)))
        fz2_1 = min(occ2.shape[2], int(math.ceil(fb.zmax / 2.0)))
        if fx1 > fx0 and fy1 > fy0:
            if fz6_1 > fz6_0:
                occ6[fx0:fx1, fy0:fy1, fz6_0:fz6_1] = False
            if fz2_1 > fz2_0:
                occ2[fx0:fx1, fy0:fy1, fz2_0:fz2_1] = False

    z6_any = np.any(occ6, axis=(0, 1))
    gz6_nonempty_limit = int(np.max(np.nonzero(z6_any)[0]) + 1) if np.any(z6_any) else occ6.shape[2]
    occ6_csum = np.zeros((occ6.shape[0] + 1, occ6.shape[1] + 1, occ6.shape[2] + 1), dtype=np.int32)
    occ6_csum[1:, 1:, 1:] = np.cumsum(np.cumsum(np.cumsum(occ6.astype(np.int32), axis=0), axis=1), axis=2)
    build_mold_counts: Dict[str, int] = {}

    # 2. Online Top-K Open-Catalog Retrieval (`< 1.5 ms`)
    ret_res = (
        _pre_ret
        if _pre_ret is not None
        else CATALOG_RETRIEVAL_INDEX.retrieve_top_k(
            occ6,
            top_k=retrieval_top_k,
            license_filter=license_filter,
            semantic_style_hint=semantic_style_hint,
        )
    )
    if part_retrieval == "open_catalog_top_k":
        kit_ids = list(ret_res.retrieved_part_ids)
    elif allowed_part_ids is not None:
        kit_ids = list(allowed_part_ids)
    else:
        kit_ids = list(ret_res.retrieved_part_ids)

    if license_filter == "cc_by_4_0_only":
        kit_ids = [
            pid
            for pid in kit_ids
            if (
                pid in EXTENDED_BRICK_CATALOG
                or (
                    CATALOG_RETRIEVAL_INDEX.get_entry(pid) is not None
                    and CATALOG_RETRIEVAL_INDEX.get_entry(pid).license_tier == "cc_by_4_0"  # type: ignore[union-attr]
                )
            )
        ]

    kit_set = set(kit_ids)
    rem_inv: Optional[Dict[str, int]] = dict(inventory_counts) if inventory_counts is not None else None
    smooth_mode = semantic_style_hint == "smooth_studless"
    ad_occ6 = occ6[:s6x, :s6y, :s6z]

    # 2b. Upfront `< 8 ms` Blind Sub-Plate Portal-Bridge & Stepped-Frustum Fast-Path
    if (
        _forced_plan is None
        and not _eval_only
        and rem_inv is None
        and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
    ):
        st_fast = try_upfront_blind_sculpted_fastpath(
            kernel=kernel,
            occ6=ad_occ6,
            kit_ids=kit_ids,
            forbidden_support_boxes=forbidden_support_boxes,
            foundation_policy=foundation_policy,
        )
        if st_fast is not None:
            if smooth_mode:
                _apply_terminal_tile_swap(st_fast, kit_set)
            mean_de00_f = 0.0
            if vertex_colors_rgb is not None:
                v_pts_f = verts_ldu if verts_ldu is not None else adaptive_target.mesh_surf_pts
                mean_de00_f = assign_perceptual_ldraw_colors(st_fast, v_pts_f, vertex_colors_rgb)
            fine_tf_f = _build_fine_target_field_from_occ6(occ6)
            distinct_f = tuple(sorted({p.part_id for p in st_fast.placed_parts.values()}))
            sculpt_f = 0
            jump_f = 0
            snot_f = 0
            tile_f = 0
            for p_f in st_fast.placed_parts.values():
                ent_f = CATALOG_RETRIEVAL_INDEX.get_entry(p_f.part_id)
                cls_f = ent_f.taxonomy_class if ent_f is not None else ""
                if cls_f in SCULPTURAL_TAXONOMY_CLASSES:
                    sculpt_f += 1
                if cls_f == "JUMPER_HALF_STUD":
                    jump_f += 1
                if cls_f == "SNOT_ANCHOR" or LDUFrame(p_f.pose.rot_id, (0, 0, 0)).z_axis[2] == 0:
                    snot_f += 1
                if cls_f == "SMOOTH_TILE" or p_f.part_id in PLATE_TO_SMOOTH_TILE_MAP.values():
                    tile_f += 1
            return (
                st_fast,
                fine_tf_f,
                OpenCatalogRealizeDiagnostics(
                    retrieval_result=ret_res,
                    distinct_parts_placed=distinct_f,
                    sculptural_parts_placed_count=sculpt_f,
                    jumper_parts_placed_count=jump_f,
                    snot_parts_placed_count=snot_f,
                    smooth_tiles_placed_count=tile_f,
                    exposed_upward_studs_count=count_exposed_upward_studs(st_fast),
                    mean_color_delta_e00=round(mean_de00_f, 4),
                ),
            )

    # 3. Phase-Aware `8 LDU` Plate-Layer + `+10 LDU` Jumper Eroded Core Synthesis
    #    Check ground sub-stud phase `(ox0, oy0)` on `CLUTCHED_BASEPLATE` (`(0,0), (10,0), (0,10), (10,10)`)
    def _cell_counts_at_phase(ox: int, oy: int) -> np.ndarray:
        """Return `(nx_s - 1, ny_s - 1, n_plates_z)` int16 sub-voxel counts (`0..64`) on `occ2`."""
        sx_0 = ox // 5
        sy_0 = oy // 5
        w_x = nx_s - 1
        w_y = ny_s - 1
        sub = occ2[sx_0 : sx_0 + 4 * w_x, sy_0 : sy_0 + 4 * w_y, : 4 * n_plates_z]
        return np.sum(
            sub.reshape(w_x, 4, w_y, 4, n_plates_z, 4), axis=(1, 3, 5), dtype=np.int16
        )

    phase_grids: Dict[Tuple[int, int], np.ndarray] = {
        (ox, oy): _cell_counts_at_phase(ox, oy)
        for ox in (0, 10)
        for oy in (0, 10)
    }

    # Detect thin-wireframe / thin-slab / narrow-blade regime purely from sub-voxel occupancy density
    all_g_phases = (
        [(0, 0), (10, 0), (0, 10), (10, 10)]
        if foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
        else [(0, 0)]
    )
    max_cg_48 = max(int(np.count_nonzero(phase_grids[ph] >= 48)) for ph in all_g_phases)
    max_cg_32 = max(int(np.count_nonzero(phase_grids[ph] >= 32)) for ph in all_g_phases)
    max_cg_16 = max(int(np.count_nonzero(phase_grids[ph] >= 16)) for ph in all_g_phases)
    ground_c24 = int(np.count_nonzero(phase_grids[(0, 0)][:, :, 0] >= 24))
    peak_c24 = max(
        (int(np.count_nonzero(phase_grids[(0, 0)][:, :, q_i] >= 24)) for q_i in range(n_plates_z)),
        default=0,
    )
    ground_c16 = int(np.count_nonzero(phase_grids[(0, 0)][:, :, 0] >= 16))
    peak_c16 = max(
        (int(np.count_nonzero(phase_grids[(0, 0)][:, :, q_i] >= 16)) for q_i in range(n_plates_z)),
        default=0,
    )
    mesh_sub_total_init = int(np.count_nonzero(occ6))
    is_legged_frame = (
        (
            (peak_c24 >= 6 and ground_c24 <= int(0.26 * peak_c24))
            or (peak_c16 >= 6 and ground_c16 <= max(1, int(0.28 * peak_c16)))
        )
        and (not forbidden_support_boxes)
    )
    is_thin_regime = (
        is_legged_frame
        or (max_cg_48 <= 4)
        or (max_cg_32 < max(8, int(0.38 * max_cg_16)))
        or (mesh_sub_total_init < 720 and max_cg_48 <= 16)
    )
    eff_gz6_sky_limit = gz6_nonempty_limit
    if not is_thin_regime and gz6_nonempty_limit >= 10:
        cnt_sky_top = int(np.count_nonzero(occ6[:, :, gz6_nonempty_limit - 1]))
        cnt_sky_bel = int(np.count_nonzero(occ6[:, :, gz6_nonempty_limit - 2]))
        if cnt_sky_bel >= 32 and cnt_sky_top * 6 <= cnt_sky_bel:
            eff_gz6_sky_limit = gz6_nonempty_limit - 1
    candidates_g_phase = all_g_phases

    # Choose ground phase `(ox_cur, oy_cur)`: on thick multi-tier meshes inspect Course 0 (`z = 0..24`)
    # so shifted upper tiers never bias the baseplate; on thin-frame / narrow-wall `.obj` meshes align the whole frame!
    # Penalize orphaned sub-voxels in `occ2` falling outside `[ox, ox + 20*w_x] x [oy, oy + 20*w_y]`.
    best_ground_phase = (0, 0)
    best_g_score = -1e9
    wx4 = 4 * (nx_s - 1)
    wy4 = 4 * (ny_s - 1)
    is_narrow_xy = min(nx_s - 1, ny_s - 1) <= 2
    for ph in candidates_g_phase:
        sx0_p = ph[0] // 5
        sy0_p = ph[1] // 5
        if is_thin_regime or is_narrow_xy:
            z_lim2 = min(occ2.shape[2], 4 * n_plates_z)
            orphan_cnt = (
                int(np.count_nonzero(occ2[:sx0_p, :, :z_lim2]))
                + int(np.count_nonzero(occ2[sx0_p + wx4 :, :, :z_lim2]))
                + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, :sy0_p, :z_lim2]))
                + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, sy0_p + wy4 :, :z_lim2]))
            )
            cg = phase_grids[ph]
            active_thin = cg >= 14
            sc = (
                float(np.sum(2 * cg[active_thin].astype(np.int32) - 24))
                - 3.5 * float(orphan_cnt)
                - (8.0 if ph != (0, 0) else 0.0)
            )
        else:
            z_lim2 = min(occ2.shape[2], 4 * min(3, n_plates_z))
            orphan_cnt = (
                int(np.count_nonzero(occ2[:sx0_p, :, :z_lim2]))
                + int(np.count_nonzero(occ2[sx0_p + wx4 :, :, :z_lim2]))
                + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, :sy0_p, :z_lim2]))
                + int(np.count_nonzero(occ2[sx0_p : sx0_p + wx4, sy0_p + wy4 :, :z_lim2]))
            )
            cg = phase_grids[ph][:, :, : min(3, n_plates_z)]
            solid = cg >= 36
            ambig = (cg >= 16) & (cg < 36)
            sc = (
                float(np.sum(2 * cg[solid].astype(np.int32) - 64))
                - 12.0 * float(np.count_nonzero(ambig))
                - 3.0 * float(orphan_cnt)
                - (4.0 if ph != (0, 0) else 0.0)
            )
        if sc > best_g_score:
            best_g_score = sc
            best_ground_phase = ph

    w_x = nx_s - 1
    w_y = ny_s - 1
    has_snot_core = any(sp in kit_set for sp in ("30414", "11211", "87087", "44728", "4070", "99780"))
    has_thin_panel_kit = any(tp in kit_set for tp in ("4865a", "23950", "30413", "93095", "4864a", "2362a", "4215a"))

    def _build_initial_core_occ(
        l_phases: Sequence[Tuple[int, int]],
        j_bridges: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        thr_use: int,
    ) -> np.ndarray:
        c_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
        for p in range(n_plates_z):
            if p in j_bridges:
                continue
            ph_use = l_phases[p]
            cg_p = phase_grids[ph_use][:, :, p]
            z6_prev_p = min(occ6.shape[2], int(math.ceil((8 * p) / 6.0)))
            z6_cur_p = min(occ6.shape[2], int(math.ceil((8 * (p + 1)) / 6.0)))
            for ix in range(w_x):
                for iy in range(w_y):
                    val_c = int(cg_p[ix, iy])
                    take_cell = False
                    if val_c >= thr_use:
                        take_cell = True
                    elif (
                        p > 0
                        and (p - 1) not in j_bridges
                        and l_phases[p - 1] == ph_use
                        and c_occ[ix, iy, p - 1]
                        and val_c >= 12
                        and z6_cur_p > z6_prev_p
                    ):
                        # Top-cap partial-plate recovery (`e.g. z=[96..100]` where `d_in >= d_ov` on `occ6`)
                        x6_c = (ix * 20 + ph_use[0]) // 5
                        y6_c = (iy * 20 + ph_use[1]) // 5
                        blk_cap = occ6[x6_c : x6_c + 4, y6_c : y6_c + 4, z6_prev_p:z6_cur_p]
                        is_roof_cap_cell = (p + 1) >= n_plates_z or int(phase_grids[ph_use][ix, iy, p + 1]) < 8
                        cap_frac_req = 0.50 if is_roof_cap_cell else max(0.50, float(thr_use) / 64.0)
                        if blk_cap.size > 0 and float(np.count_nonzero(blk_cap)) >= cap_frac_req * float(blk_cap.size):
                            take_cell = True
                    elif is_thin_regime and not is_legged_frame and thr_use <= 32 and val_c >= 8:
                        col_z = phase_grids[ph_use][ix, iy, :]
                        has_solid_below = p > 0 and bool(np.any(col_z[:p] >= 24))
                        has_solid_above = (p + 1) < n_plates_z and bool(np.any(col_z[p + 1 :] >= 24))
                        if has_solid_below and has_solid_above:
                            take_cell = True
                    if not take_cell and is_thin_regime and thr_use <= 32 and p >= 3 and val_c >= 6:
                        # Capture elevated horizontal thin slabs (`e.g. thin tabletops`) including perimeter/corner cells
                        sx0_c = (ix * 20 + ph_use[0]) // 5
                        sy0_c = (iy * 20 + ph_use[1]) // 5
                        sub_xy = occ2[sx0_c : sx0_c + 4, sy0_c : sy0_c + 4, 4 * p : 4 * (p + 1)]
                        x_span_5 = int(np.count_nonzero(np.any(sub_xy, axis=(1, 2))))
                        y_span_5 = int(np.count_nonzero(np.any(sub_xy, axis=(0, 2))))
                        nbr_12 = sum(
                            1
                            for dx_n, dy_n in ((1, 0), (-1, 0), (0, 1), (0, -1))
                            if 0 <= ix + dx_n < w_x
                            and 0 <= iy + dy_n < w_y
                            and int(cg_p[ix + dx_n, iy + dy_n]) >= 8
                        )
                        if x_span_5 >= 2 and y_span_5 >= 2 and nbr_12 >= 1 and int(np.count_nonzero(cg_p >= 8)) >= 6:
                            take_cell = True
                    if take_cell:
                        wb = LDUBox(
                            ix * 20 + ph_use[0],
                            iy * 20 + ph_use[1],
                            p * 8,
                            (ix + 1) * 20 + ph_use[0],
                            (iy + 1) * 20 + ph_use[1],
                            (p + 1) * 8,
                        )
                        if not _box_hits_forbidden(wb, forbidden_support_boxes):
                            c_occ[ix, iy, p] = True

        if is_thin_regime:
            for p_sl in range(5, n_plates_z - 1):
                if p_sl in j_bridges or (p_sl + 1) in j_bridges or l_phases[p_sl] != l_phases[p_sl + 1]:
                    continue
                ph_sl = l_phases[p_sl]
                n_a = int(np.count_nonzero(c_occ[:, :, p_sl]))
                n_b = int(np.count_nonzero(c_occ[:, :, p_sl + 1]))
                if n_a >= 4 and n_b >= 4:
                    def _layer_occ6_mask(p_eval: int) -> np.ndarray:
                        m6 = np.zeros_like(occ6)
                        z6_0 = max(0, int(math.floor((p_eval * 8) / 6.0)))
                        z6_1 = min(occ6.shape[2], int(math.ceil(((p_eval + 1) * 8) / 6.0)))
                        for ix_e in range(w_x):
                            for iy_e in range(w_y):
                                if c_occ[ix_e, iy_e, p_eval]:
                                    x6_0 = (ix_e * 20 + ph_sl[0]) // 5
                                    y6_0 = (iy_e * 20 + ph_sl[1]) // 5
                                    m6[x6_0 : x6_0 + 4, y6_0 : y6_0 + 4, z6_0:z6_1] = True
                        return m6

                    ma = _layer_occ6_mask(p_sl)
                    mb = _layer_occ6_mask(p_sl + 1)
                    mab = ma | mb
                    net_a = 2 * int(np.count_nonzero(ma & occ6)) - int(np.count_nonzero(ma))
                    net_b = 2 * int(np.count_nonzero(mb & occ6)) - int(np.count_nonzero(mb))
                    net_ab = 2 * int(np.count_nonzero(mab & occ6)) - int(np.count_nonzero(mab))
                    if max(net_a, net_b) >= net_ab + 16:
                        if net_a >= net_b:
                            c_occ[:, :, p_sl + 1] = False
                        else:
                            c_occ[:, :, p_sl] = False
        return c_occ

    # Pure-NumPy `(layer_phase, jumper_bridge_layers, core_cell_thresh)` model selector directly on `occ6` (`< 3 ms`)
    default_thr = 22 if is_thin_regime else (24 if max_cg_48 < 14 else 30)
    if (
        not forbidden_support_boxes
        and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
        and (is_thin_regime or is_narrow_xy)
    ):
        cand_phases_sel = all_g_phases
    else:
        cand_phases_sel = [best_ground_phase]

    if is_thin_regime:
        cand_threshs_sel = [16, 22, 30]
    elif not forbidden_support_boxes:
        cand_threshs_sel = [22, 28, 33]
    else:
        cand_threshs_sel = [default_thr]

    def _est_core_occ6_iou(
        c_arr: np.ndarray,
        l_phases_c: Sequence[Tuple[int, int]],
        j_bridges_c: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> float:
        co6 = np.zeros_like(occ6)
        for p_c0 in range(n_plates_z):
            if p_c0 in j_bridges_c or not np.any(c_arr[:, :, p_c0]):
                continue
            ph_c0 = l_phases_c[p_c0]
            z6_c0 = max(0, (p_c0 * 8) // 6)
            z6_c1 = min(occ6.shape[2], int(math.ceil(((p_c0 + 1) * 8) / 6.0)))
            for ix_c0 in range(w_x):
                for iy_c0 in range(w_y):
                    if c_arr[ix_c0, iy_c0, p_c0]:
                        x6_c0 = (ix_c0 * 20 + ph_c0[0]) // 5
                        y6_c0 = (iy_c0 * 20 + ph_c0[1]) // 5
                        co6[x6_c0 : x6_c0 + 4, y6_c0 : y6_c0 + 4, z6_c0:z6_c1] = True
        m_sub = max(1, int(np.count_nonzero(occ6)))
        c_in = int(np.count_nonzero(co6 & occ6))
        c_ov = int(np.count_nonzero(co6 & (~occ6)))
        return float(c_in) / float(max(1, m_sub + c_ov))

    def _fast_est_plan_occ6_iou(
        l_phases_e: Sequence[Tuple[int, int]],
        j_bridges_e: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        thr_e: int,
    ) -> float:
        c_e = _build_initial_core_occ(l_phases_e, j_bridges_e, thr_e)
        if is_thin_regime:
            marg_r = 0.20 if has_thin_panel_kit else 0.35
        else:
            iou_c0 = _est_core_occ6_iou(c_e, l_phases_e, j_bridges_e)
            marg_r = max(0.455, iou_c0 / (1.0 + iou_c0))
        for p_pr in range(n_plates_z - 1, 0, -1):
            if p_pr in j_bridges_e or (p_pr - 1) in j_bridges_e or l_phases_e[p_pr] != l_phases_e[p_pr - 1]:
                continue
            ph_e = l_phases_e[p_pr]
            z6_p0 = min(occ6.shape[2], int(math.ceil((8 * p_pr) / 6.0)))
            z6_p1 = min(occ6.shape[2], int(math.ceil((8 * (p_pr + 1)) / 6.0)))
            if z6_p1 <= z6_p0:
                continue
            for ix_p in range(w_x):
                for iy_p in range(w_y):
                    if c_e[ix_p, iy_p, p_pr] and c_e[ix_p, iy_p, p_pr - 1] and not (
                        (p_pr + 1) < n_plates_z and np.any(c_e[ix_p, iy_p, p_pr + 1 :])
                    ):
                        x6_p = (ix_p * 20 + ph_e[0]) // 5
                        y6_p = (iy_p * 20 + ph_e[1]) // 5
                        bm = occ6[x6_p : x6_p + 4, y6_p : y6_p + 4, z6_p0:z6_p1]
                        if bm.size > 0 and float(np.count_nonzero(bm)) < marg_r * float(bm.size):
                            c_e[ix_p, iy_p, p_pr] = False
        if has_snot_core and not is_thin_regime and not forbidden_support_boxes:
            for c_l in range((n_plates_z // 3) - 1, -1, -1):
                p0_e = 3 * c_l
                if any((p0_e + q_e) in j_bridges_e for q_e in range(min(4, n_plates_z - p0_e))):
                    continue
                ph_e = l_phases_e[p0_e]
                if l_phases_e[p0_e + 1] != ph_e or l_phases_e[p0_e + 2] != ph_e:
                    continue
                carve_e: Set[Tuple[int, int]] = set()
                for ix_e in range(w_x):
                    for iy_e in range(w_y):
                        if not np.any(c_e[ix_e, iy_e, p0_e : p0_e + 3]):
                            continue
                        if (p0_e + 3) < n_plates_z and np.any(c_e[ix_e, iy_e, p0_e + 3 :]):
                            continue
                        sx_e = (ix_e * 20 + ph_e[0]) // 5
                        sy_e = (iy_e * 20 + ph_e[1]) // 5
                        bl = occ2[sx_e : sx_e + 4, sy_e : sy_e + 4, 4 * p0_e : 4 * (p0_e + 3)]
                        for dx_o, dy_o in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ox_i, oy_i = ix_e + dx_o, iy_e + dy_o
                            in_x, in_y = ix_e - dx_o, iy_e - dy_o
                            if 0 <= ox_i < w_x and 0 <= oy_i < w_y and np.any(c_e[ox_i, oy_i, p0_e : p0_e + 3]):
                                continue
                            if not (0 <= in_x < w_x and 0 <= in_y < w_y and np.all(c_e[in_x, in_y, p0_e : p0_e + 3])):
                                continue
                            if (in_x, in_y) in carve_e:
                                continue
                            if dx_o == 1:
                                op0 = int(np.count_nonzero(bl[2:, :, :4]))
                                oc, ic = int(np.count_nonzero(bl[2:, :, :])), int(np.count_nonzero(bl[:2, :, :]))
                            elif dx_o == -1:
                                op0 = int(np.count_nonzero(bl[:2, :, :4]))
                                oc, ic = int(np.count_nonzero(bl[:2, :, :])), int(np.count_nonzero(bl[2:, :, :]))
                            elif dy_o == 1:
                                op0 = int(np.count_nonzero(bl[:, 2:, :4]))
                                oc, ic = int(np.count_nonzero(bl[:, 2:, :])), int(np.count_nonzero(bl[:, :2, :]))
                            else:
                                op0 = int(np.count_nonzero(bl[:, :2, :4]))
                                oc, ic = int(np.count_nonzero(bl[:, :2, :])), int(np.count_nonzero(bl[:, 2:, :]))
                            if op0 <= 6 and oc <= 18 and ic >= 28:
                                carve_e.add((ix_e, iy_e))
                                break
                for cx_e, cy_e in carve_e:
                    c_e[cx_e, cy_e, p0_e : p0_e + 3] = False

        m6_e = np.zeros_like(occ6)
        j_top_e: Dict[int, np.ndarray] = {}
        for p_e in range(n_plates_z):
            z6_a = max(0, (p_e * 8) // 6)
            z6_b = min(occ6.shape[2], int(math.ceil(((p_e + 1) * 8) / 6.0)))
            if p_e in j_bridges_e:
                ph_f_j, ph_t_j = j_bridges_e[p_e]
                sh_x = ph_t_j[0] != ph_f_j[0]
                low_j = (
                    j_top_e[p_e - 1]
                    if (p_e - 1) in j_top_e
                    else (c_e[:, :, p_e - 1] if p_e > 0 else np.zeros((w_x, w_y), dtype=bool))
                )
                p_u_e = p_e + 1
                while p_u_e < n_plates_z and p_u_e in j_bridges_e:
                    p_u_e += 1
                if p_u_e < n_plates_z:
                    ph_fin_e = l_phases_e[p_u_e]
                    raw_u_e = c_e[:, :, p_u_e]
                    if ph_fin_e == ph_t_j:
                        up_j = raw_u_e
                    else:
                        up_j = np.zeros((w_x, w_y), dtype=bool)
                        uxs_e, uys_e = np.nonzero(raw_u_e)
                        uys_list_e = sorted(set(uys_e.tolist()))
                        trim_uy_e: Set[int] = set()
                        if sh_x and len(uys_list_e) >= 4:
                            cg_le = phase_grids[ph_f_j][:, :, p_e]
                            if ((uys_list_e[0] * 20 + ph_fin_e[1] - ph_t_j[1]) // 20) == 0 and int(np.max(cg_le[:, 0])) < 54:
                                trim_uy_e.add(uys_list_e[0])
                            if (((uys_list_e[-1] * 20 + ph_fin_e[1] - ph_t_j[1]) // 20) + 1) == (w_y - 1) and int(np.max(cg_le[:, w_y - 1])) < 54:
                                trim_uy_e.add(uys_list_e[-1])
                        for uxf_e, uyf_e in zip(uxs_e.tolist(), uys_e.tolist()):
                            if sh_x:
                                if uyf_e in trim_uy_e:
                                    continue
                                iym_e = (uyf_e * 20 + ph_fin_e[1] - ph_t_j[1]) // 20
                                for iyc_e in (iym_e, iym_e + 1):
                                    if 0 <= iyc_e < w_y:
                                        up_j[uxf_e, iyc_e] = True
                            else:
                                ixm_e = (uxf_e * 20 + ph_fin_e[0] - ph_t_j[0]) // 20
                                for ixc_e in (ixm_e, ixm_e + 1):
                                    if 0 <= ixc_e < w_x:
                                        up_j[ixc_e, uyf_e] = True
                else:
                    up_j = np.zeros((w_x, w_y), dtype=bool)
                top_j = np.zeros((w_x, w_y), dtype=bool)
                used_l_j = np.zeros((w_x, w_y), dtype=bool)
                for ux_j in range(w_x):
                    for uy_j in range(w_y):
                        if not up_j[ux_j, uy_j]:
                            continue
                        if sh_x:
                            w_cx = ux_j * 20 + ph_t_j[0] + 10
                            ix_l = (w_cx - 10 - ph_f_j[0]) // 20
                            if (
                                0 <= ix_l
                                and (ix_l + 1) < w_x
                                and not used_l_j[ix_l, uy_j]
                                and not used_l_j[ix_l + 1, uy_j]
                                and low_j[ix_l, uy_j]
                                and low_j[ix_l + 1, uy_j]
                            ):
                                x6_j = (ix_l * 20 + ph_f_j[0]) // 5
                                y6_j = (uy_j * 20 + ph_f_j[1]) // 5
                                m6_e[x6_j : x6_j + 8, y6_j : y6_j + 4, z6_a:z6_b] = True
                                used_l_j[ix_l, uy_j] = True
                                used_l_j[ix_l + 1, uy_j] = True
                                top_j[ux_j, uy_j] = True
                        else:
                            w_cy = uy_j * 20 + ph_t_j[1] + 10
                            iy_l = (w_cy - 10 - ph_f_j[1]) // 20
                            if (
                                0 <= iy_l
                                and (iy_l + 1) < w_y
                                and not used_l_j[ux_j, iy_l]
                                and not used_l_j[ux_j, iy_l + 1]
                                and low_j[ux_j, iy_l]
                                and low_j[ux_j, iy_l + 1]
                            ):
                                x6_j = (ux_j * 20 + ph_f_j[0]) // 5
                                y6_j = (iy_l * 20 + ph_f_j[1]) // 5
                                m6_e[x6_j : x6_j + 4, y6_j : y6_j + 8, z6_a:z6_b] = True
                                used_l_j[ux_j, iy_l] = True
                                used_l_j[ux_j, iy_l + 1] = True
                                top_j[ux_j, uy_j] = True
                # Include terrace smooth-tile caps on exposed lower-tier shoulders
                cg_ter_e = phase_grids[ph_f_j][:, :, p_e]
                for tx_e in range(w_x):
                    for ty_e in range(w_y):
                        if low_j[tx_e, ty_e] and not used_l_j[tx_e, ty_e] and int(cg_ter_e[tx_e, ty_e]) >= 42:
                            x6_t = (tx_e * 20 + ph_f_j[0]) // 5
                            y6_t = (ty_e * 20 + ph_f_j[1]) // 5
                            t_sl_e = occ6[x6_t : x6_t + 4, y6_t : y6_t + 4, z6_a:z6_b]
                            if t_sl_e.size > 0 and 2 * int(np.count_nonzero(t_sl_e)) >= int(t_sl_e.size):
                                m6_e[x6_t : x6_t + 4, y6_t : y6_t + 4, z6_a:z6_b] = True
                j_top_e[p_e] = top_j
                continue
            if not np.any(c_e[:, :, p_e]):
                continue
            ph_e = l_phases_e[p_e]
            for ix_e in range(w_x):
                for iy_e in range(w_y):
                    x6_e = (ix_e * 20 + ph_e[0]) // 5
                    y6_e = (iy_e * 20 + ph_e[1]) // 5
                    if c_e[ix_e, iy_e, p_e]:
                        m6_e[x6_e : x6_e + 4, y6_e : y6_e + 4, z6_a:z6_b] = True
                    elif (
                        is_thin_regime
                        and has_thin_panel_kit
                        and p_e >= 2
                        and np.any(c_e[ix_e, iy_e, :p_e])
                        and x6_e + 4 <= occ6.shape[0]
                        and y6_e + 4 <= occ6.shape[1]
                    ):
                        # Truthfully score 5 LDU perimeter thin-panel walls above supported base cells
                        for wx0, wx1, wy0, wy1 in (
                            (x6_e, x6_e + 1, y6_e, y6_e + 4),
                            (x6_e + 3, x6_e + 4, y6_e, y6_e + 4),
                            (x6_e, x6_e + 4, y6_e, y6_e + 1),
                            (x6_e, x6_e + 4, y6_e + 3, y6_e + 4),
                        ):
                            w_sl = occ6[wx0:wx1, wy0:wy1, z6_a:z6_b]
                            w_in = int(np.count_nonzero(w_sl))
                            w_ov = int(w_sl.size) - w_in
                            if w_in >= 2 and w_in * 3 >= w_ov:
                                m6_e[wx0:wx1, wy0:wy1, z6_a:z6_b] = True
                                break
        if has_snot_core and not is_thin_regime:
            for c_l in range(n_plates_z // 3):
                p0_e = 3 * c_l
                if any((p0_e + q_e) in j_bridges_e for q_e in range(3)):
                    continue
                ph_e = l_phases_e[p0_e]
                if l_phases_e[p0_e + 1] != ph_e or l_phases_e[p0_e + 2] != ph_e:
                    continue
                z6_f0 = max(0, (p0_e * 8) // 6)
                z6_f1 = min(occ6.shape[2], (p0_e * 8 + 24) // 6)
                for ix_e in range(w_x):
                    for iy_e in range(w_y):
                        if not np.all(c_e[ix_e, iy_e, p0_e : p0_e + 3]):
                            continue
                        fx0_e, fx1_e = ix_e * 20 + ph_e[0], (ix_e + 1) * 20 + ph_e[0]
                        fy0_e, fy1_e = iy_e * 20 + ph_e[1], (iy_e + 1) * 20 + ph_e[1]
                        for dx_o, dy_o, mx0, mx1, my0, my1 in (
                            (1, 0, fx1_e // 5, fx1_e // 5 + 2, fy0_e // 5, fy1_e // 5),
                            (-1, 0, fx0_e // 5 - 2, fx0_e // 5, fy0_e // 5, fy1_e // 5),
                            (0, 1, fx0_e // 5, fx1_e // 5, fy1_e // 5, fy1_e // 5 + 2),
                            (0, -1, fx0_e // 5, fx1_e // 5, fy0_e // 5 - 2, fy0_e // 5),
                        ):
                            ox_i, oy_i = ix_e + dx_o, iy_e + dy_o
                            if 0 <= ox_i < w_x and 0 <= oy_i < w_y and np.any(c_e[ox_i, oy_i, p0_e : p0_e + 3]):
                                continue
                            if 0 <= mx0 and 0 <= my0 and mx1 <= occ6.shape[0] and my1 <= occ6.shape[1]:
                                ms_e = occ6[mx0:mx1, my0:my1, z6_f0:z6_f1]
                                if ms_e.size > 0 and int(np.count_nonzero(ms_e)) >= int(0.40 * ms_e.size):
                                    m6_e[mx0:mx1, my0:my1, z6_f0:z6_f1] = True
        inter_e = int(np.count_nonzero(m6_e & occ6))
        union_e = int(np.count_nonzero(m6_e | occ6))
        return float(inter_e) / float(max(1, union_e))

    if _forced_plan is not None:
        layer_phase = list(_forced_plan[0])
        jumper_bridge_layers = dict(_forced_plan[1])
        core_cell_thresh = int(_forced_plan[2])
    else:
        best_sel_iou = -1.0
        core_cell_thresh = default_thr
        for ph_try in cand_phases_sel:
            for thr_try in cand_threshs_sel:
                est_iou = _fast_est_plan_occ6_iou([ph_try] * n_plates_z, {}, thr_try)
                if ph_try == best_ground_phase:
                    est_iou += 2.0e-4
                if thr_try == default_thr:
                    est_iou += 1.0e-4
                if est_iou > best_sel_iou:
                    best_sel_iou = est_iou
                    best_ground_phase = ph_try
                    core_cell_thresh = thr_try

        layer_phase = [best_ground_phase] * n_plates_z
        jumper_bridge_layers = {}

    core_occ = _build_initial_core_occ(layer_phase, jumper_bridge_layers, core_cell_thresh)
    pier_protected = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
    pillar_added_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)

    # Pre-protect pier columns bordering forbidden architectural openings so `inv_arch_molds` never carves portal piers
    if forbidden_support_boxes:
        for fb in forbidden_support_boxes:
            p_fb_max = min(n_plates_z, int(math.ceil(fb.zmax / 8.0)) + 1)
            for p_fb in range(p_fb_max):
                ph_fb = layer_phase[p_fb]
                fx0_i = max(0, (fb.xmin - ph_fb[0]) // 20 - 1)
                fx1_i = min(w_x, (fb.xmax - ph_fb[0] + 19) // 20 + 1)
                fy0_i = max(0, (fb.ymin - ph_fb[1]) // 20 - 1)
                fy1_i = min(w_y, (fb.ymax - ph_fb[1] + 19) // 20 + 1)
                pier_protected[fx0_i:fx1_i, fy0_i:fy1_i, p_fb] = True

    # Reserve bottom-up Open-Catalog `INVERTED_SLOPE` & `ARCH` molds (`3660, 3665, 4287c,
    # 3659, 15254, 3308`, etc. with `num_top_studs >= 1`) on outward-flaring overhangs BEFORE dropping pillars!
    inv_reserved_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
    planned_inv_by_z: Dict[int, List[Tuple[str, LDUFrame]]] = {}
    inv_arch_molds = [
        pid
        for pid in kit_ids
        if CATALOG_RETRIEVAL_INDEX.get_entry(pid) is not None
        and CATALOG_RETRIEVAL_INDEX.get_entry(pid).taxonomy_class in ("INVERTED_SLOPE", "ARCH")  # type: ignore[union-attr]
        and CATALOG_RETRIEVAL_INDEX.get_entry(pid).num_top_studs >= 1  # type: ignore[union-attr]
        and len(get_brick_primitive(pid).hull_boxes) >= 2
    ]
    if inv_arch_molds:
        has_openings_inv = bool(forbidden_support_boxes)
        cand_p0_inv = (
            list(range(0, max(0, n_plates_z - 2)))
            if (foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE and has_openings_inv)
            else [3 * c_i for c_i in range(1, n_plates_z // 3)]
        )
        for p0_i in cand_p0_inv:
            if (p0_i > 0 and (p0_i - 1) in jumper_bridge_layers) or any(
                (p0_i + q_i) in jumper_bridge_layers for q_i in range(3) if (p0_i + q_i) < n_plates_z
            ):
                continue
            ph_i = layer_phase[p0_i]
            if p0_i > 0 and layer_phase[p0_i - 1] != ph_i:
                continue
            z0_i = p0_i * 8
            inv_cands_course: List[Tuple[float, str, LDUFrame, List[Tuple[int, int, int]], List[Tuple[int, int, int]]]] = []
            seen_inv_poses: Set[Tuple[str, int, Tuple[int, int, int]]] = set()
            sup_xy_i = (
                np.ones((w_x, w_y), dtype=bool)
                if p0_i == 0
                else core_occ[:, :, p0_i - 1]
            )
            if not sup_xy_i.any():
                continue
            # Probe perimeter cells of `sup_xy_i` (or cells adjacent to `core_occ[:,:,p0_i]` portals when `p0_i == 0`)
            ref_perim_xy = core_occ[:, :, p0_i] if p0_i == 0 else sup_xy_i
            for ix_a in range(w_x):
                for iy_a in range(w_y):
                    if not ref_perim_xy[ix_a, iy_a]:
                        continue
                    is_perim_a = any(
                        not (0 <= ix_a + dx_a < w_x and 0 <= iy_a + dy_a < w_y and ref_perim_xy[ix_a + dx_a, iy_a + dy_a])
                        for dx_a, dy_a in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
                    if not is_perim_a:
                        continue
                    px_a = ix_a * 20 + ph_i[0] + 10
                    py_a = iy_a * 20 + ph_i[1] + 10
                    for pid_ia in inv_arch_molds:
                        if rem_inv is not None and rem_inv.get(pid_ia, 0) <= 0:
                            continue
                        prim_ia = get_brick_primitive(pid_ia)
                        ent_ia = CATALOG_RETRIEVAL_INDEX.get_entry(pid_ia)
                        is_arch_mold = ent_ia is not None and ent_ia.taxonomy_class == "ARCH"
                        if p0_i == 0 and not is_arch_mold:
                            continue
                        if (p0_i % 3) != 0 and not (has_openings_inv and is_arch_mold):
                            continue
                        for rid_ia, dx_ia, dy_ia, dz_ia, rboxes_ia in _fast_upward_stud_templates(pid_ia):
                            pos_ia = (px_a + dx_ia, py_a + dy_ia, z0_i + dz_ia)
                            ikey = (pid_ia, rid_ia, pos_ia)
                            if ikey in seen_inv_poses:
                                continue
                            seen_inv_poses.add(ikey)
                            w_zmin = min(b[2] for b in rboxes_ia) + z0_i
                            w_zmax = max(b[5] for b in rboxes_ia) + z0_i
                            h_ldu_ia = w_zmax - w_zmin
                            if w_zmin != z0_i or h_ldu_ia <= 0 or (h_ldu_ia % 8) != 0:
                                continue
                            h_pl = h_ldu_ia // 8
                            if p0_i + h_pl > n_plates_z:
                                continue
                            if any(
                                layer_phase[p0_i + q_h] != ph_i or (p0_i + q_h) in jumper_bridge_layers
                                for q_h in range(h_pl)
                            ):
                                continue
                            w_xmin = min(b[0] for b in rboxes_ia) + px_a
                            w_xmax = max(b[3] for b in rboxes_ia) + px_a
                            w_ymin = min(b[1] for b in rboxes_ia) + py_a
                            w_ymax = max(b[4] for b in rboxes_ia) + py_a
                            if (
                                (w_xmin - ph_i[0]) % 20 != 0
                                or (w_xmax - ph_i[0]) % 20 != 0
                                or (w_ymin - ph_i[1]) % 20 != 0
                                or (w_ymax - ph_i[1]) % 20 != 0
                            ):
                                continue
                            rx0_i = (w_xmin - ph_i[0]) // 20
                            rx1_i = (w_xmax - ph_i[0]) // 20
                            ry0_i = (w_ymin - ph_i[1]) // 20
                            ry1_i = (w_ymax - ph_i[1]) // 20
                            if rx0_i < 0 or ry0_i < 0 or rx1_i > w_x or ry1_i > w_y:
                                continue
                            if inv_reserved_occ[rx0_i:rx1_i, ry0_i:ry1_i, p0_i : p0_i + h_pl].any():
                                continue
                            pose_ia = LDUFrame(rid_ia, pos_ia)
                            # Verify ALL bottom anti-studs rest on `sup_xy_i` (and `core_occ[:,:,p0_i]` if `p0_i == 0`) and >= 1 top stud bridges/overhangs!
                            all_bot_sup = True
                            n_bot = 0
                            top_stud_cells: List[Tuple[int, int, int]] = []
                            has_overhang = False
                            for pt_ia in prim_ia.ports:
                                wf_pt = pose_ia.compose(pt_ia.local_frame)
                                if pt_ia.polarity == PortPolarity.ANTI_STUD and wf_pt.z_axis == (0, 0, 1) and wf_pt.pos[2] == z0_i:
                                    bx_p = (wf_pt.pos[0] - ph_i[0]) // 20
                                    by_p = (wf_pt.pos[1] - ph_i[1]) // 20
                                    bot_ok = (
                                        (0 <= bx_p < w_x and 0 <= by_p < w_y and core_occ[bx_p, by_p, 0])
                                        if p0_i == 0
                                        else (0 <= bx_p < w_x and 0 <= by_p < w_y and sup_xy_i[bx_p, by_p])
                                    )
                                    if not bot_ok:
                                        all_bot_sup = False
                                        break
                                    n_bot += 1
                                elif pt_ia.polarity == PortPolarity.STUD and wf_pt.z_axis == (0, 0, 1) and wf_pt.pos[2] == w_zmax:
                                    tx_p = (wf_pt.pos[0] - ph_i[0]) // 20
                                    ty_p = (wf_pt.pos[1] - ph_i[1]) // 20
                                    if 0 <= tx_p < w_x and 0 <= ty_p < w_y:
                                        top_stud_cells.append((tx_p, ty_p, p0_i + h_pl - 1))
                                        if (not core_occ[tx_p, ty_p, 0]) if p0_i == 0 else (not sup_xy_i[tx_p, ty_p]):
                                            has_overhang = True
                            if not all_bot_sup or n_bot == 0 or not has_overhang:
                                continue
                            if (
                                is_thin_regime
                                and len(top_stud_cells) < (rx1_i - rx0_i) * (ry1_i - ry0_i)
                            ):
                                uz_i0 = min(occ6.shape[2], int(math.ceil(w_zmax / 6.0)))
                                uz_i1 = min(occ6.shape[2], uz_i0 + 4)
                                sx_i0 = max(0, w_xmin // 5)
                                sx_i1 = min(occ6.shape[0], w_xmax // 5)
                                sy_i0 = max(0, w_ymin // 5)
                                sy_i1 = min(occ6.shape[1], w_ymax // 5)
                                if uz_i1 > uz_i0 and int(np.count_nonzero(occ6[sx_i0:sx_i1, sy_i0:sy_i1, uz_i0:uz_i1])) >= 3:
                                    continue
                            wbs_ia = tuple(
                                LDUBox(b[0] + px_a, b[1] + py_a, b[2] + z0_i, b[3] + px_a, b[4] + py_a, b[5] + z0_i)
                                for b in rboxes_ia
                            )
                            if any(_box_hits_forbidden(wb_k, forbidden_support_boxes) for wb_k in wbs_ia):
                                continue
                            # Compare `occ6` net sub-voxel gain (`2*d_in - tot`) of `wbs_ia` vs. current `core_occ`
                            sx0_q = max(0, w_xmin // 5)
                            sx1_q = min(occ6.shape[0], w_xmax // 5)
                            sy0_q = max(0, w_ymin // 5)
                            sy1_q = min(occ6.shape[1], w_ymax // 5)
                            sz0_q = max(0, z0_i // 6)
                            sz1_q = min(occ6.shape[2], int(math.ceil(w_zmax / 6.0)))
                            if sx1_q <= sx0_q or sy1_q <= sy0_q or sz1_q <= sz0_q:
                                continue
                            tgt_blk = occ6[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                            ia_blk = np.zeros_like(tgt_blk)
                            for wb_k in wbs_ia:
                                kx0 = max(sx0_q, int(math.floor(wb_k.xmin / 5.0))) - sx0_q
                                kx1 = min(sx1_q, int(math.ceil(wb_k.xmax / 5.0))) - sx0_q
                                ky0 = max(sy0_q, int(math.floor(wb_k.ymin / 5.0))) - sy0_q
                                ky1 = min(sy1_q, int(math.ceil(wb_k.ymax / 5.0))) - sy0_q
                                kz0 = max(sz0_q, int(math.floor(wb_k.zmin / 6.0))) - sz0_q
                                kz1 = min(sz1_q, int(math.ceil(wb_k.zmax / 6.0))) - sz0_q
                                if kx1 > kx0 and ky1 > ky0 and kz1 > kz0:
                                    ia_blk[kx0:kx1, ky0:ky1, kz0:kz1] = True
                            core_blk = np.zeros_like(tgt_blk)
                            for rx_q in range(rx0_i, rx1_i):
                                for ry_q in range(ry0_i, ry1_i):
                                    for pq in range(p0_i, p0_i + h_pl):
                                        if core_occ[rx_q, ry_q, pq]:
                                            cx0_q = max(sx0_q, (rx_q * 20 + ph_i[0]) // 5) - sx0_q
                                            cx1_q = min(sx1_q, ((rx_q + 1) * 20 + ph_i[0]) // 5) - sx0_q
                                            cy0_q = max(sy0_q, (ry_q * 20 + ph_i[1]) // 5) - sy0_q
                                            cy1_q = min(sy1_q, ((ry_q + 1) * 20 + ph_i[1]) // 5) - sy0_q
                                            cz0_q = max(sz0_q, int(math.floor((pq * 8) / 6.0))) - sz0_q
                                            cz1_q = min(sz1_q, int(math.ceil(((pq + 1) * 8) / 6.0))) - sz0_q
                                            if cx1_q > cx0_q and cy1_q > cy0_q and cz1_q > cz0_q:
                                                core_blk[cx0_q:cx1_q, cy0_q:cy1_q, cz0_q:cz1_q] = True
                            ia_in = int(np.count_nonzero(ia_blk & tgt_blk))
                            ia_tot = int(np.count_nonzero(ia_blk))
                            co_in = int(np.count_nonzero(core_blk & tgt_blk))
                            co_tot = int(np.count_nonzero(core_blk))
                            ia_net = 2 * ia_in - ia_tot
                            co_net = 2 * co_in - co_tot
                            exact_tie_consolidation = (
                                ia_in >= 12
                                and ia_in >= co_in
                                and (ia_tot - ia_in) <= (co_tot - co_in)
                            )
                            if ia_net > 4 and (ia_net >= co_net + 2 or exact_tie_consolidation):
                                res_cells = [
                                    (rx_q, ry_q, pq)
                                    for rx_q in range(rx0_i, rx1_i)
                                    for ry_q in range(ry0_i, ry1_i)
                                    for pq in range(p0_i, p0_i + h_pl)
                                ]
                                used_inv_already = any(
                                    it_iv[0] == pid_ia for v_iv in planned_inv_by_z.values() for it_iv in v_iv
                                )
                                sc_ia = (
                                    float(ia_net - co_net)
                                    + (2.5 if is_arch_mold else 0.8)
                                    + (3.8 if not used_inv_already else 0.0)
                                    + 0.85 * float(rx1_i - rx0_i) * float(ry1_i - ry0_i)
                                )
                                inv_cands_course.append((sc_ia, pid_ia, pose_ia, res_cells, top_stud_cells))

            inv_cands_course.sort(key=lambda x: (-x[0], x[1], x[2].rot_id, x[2].pos))
            for _, pid_ch, pose_ch, r_cells, t_cells in inv_cands_course:
                if rem_inv is not None and rem_inv.get(pid_ch, 0) <= 0:
                    continue
                if any(inv_reserved_occ[rc[0], rc[1], rc[2]] for rc in r_cells):
                    continue
                for rc in r_cells:
                    inv_reserved_occ[rc[0], rc[1], rc[2]] = True
                    core_occ[rc[0], rc[1], rc[2]] = False
                    pier_protected[rc[0], rc[1], : rc[2] + 1] = True
                for tc in t_cells:
                    core_occ[tc[0], tc[1], tc[2]] = True
                    pier_protected[tc[0], tc[1], tc[2]] = True
                planned_inv_by_z.setdefault(z0_i, []).append((pid_ch, pose_ch))
                if rem_inv is not None:
                    rem_inv[pid_ch] = rem_inv.get(pid_ch, 0) - 1

    # Ensure every elevated component (`e.g. thin tabletops & canopies`) has minimal vertical pier support
    pier_cover_radius = 4 if is_legged_frame else (3 if is_thin_regime else 2)
    if support_policy == "minimal_permanent_pillars":
        for p in range(n_plates_z - 1, 0, -1):
            if p in jumper_bridge_layers or not np.any(core_occ[:, :, p]):
                continue
            p_below = p - 1
            while p_below > 0 and p_below in jumper_bridge_layers:
                p_below -= 1
            ph_p = layer_phase[p]
            vis = np.zeros((w_x, w_y), dtype=bool)
            for sx_i in range(w_x):
                for sy_i in range(w_y):
                    if not core_occ[sx_i, sy_i, p] or vis[sx_i, sy_i]:
                        continue
                    comp: List[Tuple[int, int]] = []
                    dq = deque([(sx_i, sy_i)])
                    vis[sx_i, sy_i] = True
                    while dq:
                        cx_i, cy_i = dq.popleft()
                        comp.append((cx_i, cy_i))
                        for dx_n, dy_n in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx_i, ny_i = cx_i + dx_n, cy_i + dy_n
                            if 0 <= nx_i < w_x and 0 <= ny_i < w_y and core_occ[nx_i, ny_i, p] and not vis[nx_i, ny_i]:
                                vis[nx_i, ny_i] = True
                                dq.append((nx_i, ny_i))

                    supported_pts: List[Tuple[int, int]] = []
                    ph_b = layer_phase[p_below]
                    for cx_i, cy_i in comp:
                        xb0 = cx_i * 20 + ph_p[0]
                        yb0 = cy_i * 20 + ph_p[1]
                        has_b = False
                        for bx_i in range(max(0, cx_i - 1), min(w_x, cx_i + 2)):
                            for by_i in range(max(0, cy_i - 1), min(w_y, cy_i + 2)):
                                if not core_occ[bx_i, by_i, p_below]:
                                    continue
                                if (
                                    min(xb0 + 20, bx_i * 20 + ph_b[0] + 20) > max(xb0, bx_i * 20 + ph_b[0])
                                    and min(yb0 + 20, by_i * 20 + ph_b[1] + 20) > max(yb0, by_i * 20 + ph_b[1])
                                ):
                                    has_b = True
                                    break
                            if has_b:
                                break
                        if has_b:
                            supported_pts.append((cx_i, cy_i))

                    # Skip extra pillars when >= 30% (`and >= 3 studs`) of the span rests on solid piers within <= 4 studs
                    # (counterbalanced cantilevers bridged by monolithic 2x6 / 1x6 / 2x8 girders!)
                    if not is_thin_regime and len(supported_pts) >= max(3, int(0.30 * len(comp))):
                        max_cant_d = max(
                            min(abs(pt[0] - sp[0]) + abs(pt[1] - sp[1]) for sp in supported_pts)
                            for pt in comp
                        )
                        if max_cant_d <= 4:
                            continue

                    # Consolidate ultra-thin corner/perimeter legs (`col_mesh_sum < 18 * p`, >3x empty overshoot per shaft)
                    # under an elevated slab (`4..26` studs) into a single optimal central shaft + plate capital!
                    if (
                        is_legged_frame
                        and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                        and 4 <= len(comp) <= 26
                        and p >= 4
                        and (
                            len(supported_pts) >= 2
                            or (
                                len(supported_pts) == 1
                                and len(comp) >= 12
                                and max(abs(pt[0] - supported_pts[0][0]) + abs(pt[1] - supported_pts[0][1]) for pt in comp) > 4
                            )
                        )
                        and p_below == p - 1
                        and layer_phase[p_below] == ph_p
                        and all(
                            not np.any(pier_protected[sx_l, sy_l, :p])
                            and sum(int(phase_grids[layer_phase[q_l]][sx_l, sy_l, q_l]) for q_l in range(p)) < 18 * p
                            for sx_l, sy_l in supported_pts
                        )
                    ):
                        for sx_l, sy_l in supported_pts:
                            core_occ[sx_l, sy_l, :p] = False
                        supported_pts = []

                    eff_pier_rad = pier_cover_radius
                    placed_thin_pedestal = False
                    while True:
                        unsup = [
                            pt
                            for pt in comp
                            if not supported_pts
                            or min(abs(pt[0] - sp[0]) + abs(pt[1] - sp[1]) for sp in supported_pts) > eff_pier_rad
                        ]
                        if not unsup:
                            break
                        best_anc: Optional[Tuple[int, int]] = None
                        best_col_sc = -1e9
                        best_col_mesh = 0
                        cx_mean = sum(pt[0] for pt in comp) / float(max(1, len(comp)))
                        cy_mean = sum(pt[1] for pt in comp) / float(max(1, len(comp)))
                        # Check column mesh support under `comp` to avoid phantom empty-air shafts (`col_mesh_sum == 0`)
                        max_comp_col_mesh = 0
                        if is_legged_frame and p >= 4:
                            for ux_m, uy_m in comp:
                                cm_s = sum(
                                    int(
                                        phase_grids[layer_phase[q_m]][
                                            max(0, min(w_x - 1, (ux_m * 20 + ph_p[0] - layer_phase[q_m][0]) // 20)),
                                            max(0, min(w_y - 1, (uy_m * 20 + ph_p[1] - layer_phase[q_m][1]) // 20)),
                                            q_m,
                                        ]
                                    )
                                    for q_m in range(p)
                                )
                                if cm_s > max_comp_col_mesh:
                                    max_comp_col_mesh = cm_s
                        prefer_center_pedestal = (
                            is_legged_frame and len(comp) <= 26 and p >= 4 and max_comp_col_mesh < 18 * p
                        )
                        for ux, uy in comp:
                            if (ux, uy) in supported_pts:
                                continue
                            clear_col = True
                            col_mesh_sum = 0
                            for q_dn in range(p):
                                ph_q = layer_phase[q_dn]
                                qx = max(0, min(w_x - 1, (ux * 20 + ph_p[0] - ph_q[0]) // 20))
                                qy = max(0, min(w_y - 1, (uy * 20 + ph_p[1] - ph_q[1]) // 20))
                                wb_q = LDUBox(
                                    qx * 20 + ph_q[0],
                                    qy * 20 + ph_q[1],
                                    q_dn * 8,
                                    (qx + 1) * 20 + ph_q[0],
                                    (qy + 1) * 20 + ph_q[1],
                                    (q_dn + 1) * 8,
                                )
                                if _box_hits_forbidden(wb_q, forbidden_support_boxes):
                                    clear_col = False
                                    break
                                col_mesh_sum += int(phase_grids[ph_q][qx, qy, q_dn])
                            if not clear_col:
                                continue
                            if is_legged_frame and max_comp_col_mesh >= 8 and col_mesh_sum == 0 and len(comp) <= 15:
                                continue
                            rad_chk = 10 if prefer_center_pedestal else eff_pier_rad
                            cov_gain = sum(
                                1 for pt_u in unsup if abs(pt_u[0] - ux) + abs(pt_u[1] - uy) <= rad_chk
                            )
                            cent_w = (
                                42.0
                                if (prefer_center_pedestal and len(comp) >= 12)
                                else (
                                    8.0
                                    if (prefer_center_pedestal and max_comp_col_mesh >= 16)
                                    else (38.0 if prefer_center_pedestal else 4.0)
                                )
                            )
                            cent_pen = cent_w * (
                                abs(float(ux) - cx_mean) + abs(float(uy) - cy_mean)
                            )
                            sc_a = (
                                (160.0 * float(cov_gain) + (18.0 if prefer_center_pedestal else 12.0) * float(col_mesh_sum) - cent_pen)
                                if is_legged_frame
                                else (24.0 * float(col_mesh_sum) + 12.0 * float(cov_gain))
                            )
                            if sc_a > best_col_sc:
                                best_col_sc = sc_a
                                best_anc = (ux, uy)
                                best_col_mesh = col_mesh_sum
                        if best_anc is None:
                            break
                        ax_i, ay_i = best_anc
                        supported_pts.append(best_anc)
                        pier_protected[ax_i, ay_i, p] = True
                        for q_dn in range(p - 1, -1, -1):
                            if q_dn in jumper_bridge_layers:
                                continue
                            ph_q = layer_phase[q_dn]
                            qx = max(0, min(w_x - 1, (ax_i * 20 + ph_p[0] - ph_q[0]) // 20))
                            qy = max(0, min(w_y - 1, (ay_i * 20 + ph_p[1] - ph_q[1]) // 20))
                            pier_protected[qx, qy, q_dn] = True
                            if not core_occ[qx, qy, q_dn]:
                                pillar_added_occ[qx, qy, q_dn] = True
                            core_occ[qx, qy, q_dn] = True
                        # When an elevated slab sits over ultra-thin legs where each `20x20` brick shaft overshoots >3x,
                        # give the shaft a compact `1x2` (`or symmetric 1x3 on >=16-stud wide tabletops`) anti-tipping plate capital at `p - 1`!
                        if (
                            is_legged_frame
                            and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                            and not placed_thin_pedestal
                            and p >= 4
                            and best_col_mesh < 18 * p
                            and (p - 1) not in jumper_bridge_layers
                            and layer_phase[p - 1] == ph_p
                        ):
                            sx_cap = 1 if (ax_i + 1 < w_x and (float(ax_i) <= cx_mean or ax_i == 0)) else (-1 if ax_i - 1 >= 0 else 0)
                            sy_cap = 1 if (ay_i + 1 < w_y and (float(ay_i) <= cy_mean or ay_i == 0)) else (-1 if ay_i - 1 >= 0 else 0)
                            if len(comp) >= 10 and w_x >= 4 and sy_cap != 0 and 0 <= ay_i + sy_cap < w_y:
                                x_c0 = int(round(cx_mean)) - 1
                                x_c0 = max(1 if w_x >= 5 else 0, min(w_x - (4 if w_x >= 5 else 3), x_c0))
                                if ax_i < x_c0:
                                    x_c0 = ax_i
                                elif ax_i > x_c0 + 2:
                                    x_c0 = ax_i - 2
                                x_c0 = max(0, min(w_x - 3, x_c0))
                                x_mid = x_c0 + 1
                                cap_offsets = [
                                    (x_c0 - ax_i, 0),
                                    (x_c0 + 1 - ax_i, 0),
                                    (x_c0 + 2 - ax_i, 0),
                                    (x_mid - ax_i, sy_cap),
                                    (0, sy_cap),
                                ]
                                cap_offsets = list(dict.fromkeys(cap_offsets))
                            elif len(comp) >= 10 and w_y >= 4 and sx_cap != 0 and 0 <= ax_i + sx_cap < w_x:
                                y_c0 = int(round(cy_mean)) - 1
                                y_c0 = max(1 if w_y >= 5 else 0, min(w_y - (4 if w_y >= 5 else 3), y_c0))
                                if ay_i < y_c0:
                                    y_c0 = ay_i
                                elif ay_i > y_c0 + 2:
                                    y_c0 = ay_i - 2
                                y_c0 = max(0, min(w_y - 3, y_c0))
                                y_mid = y_c0 + 1
                                cap_offsets = [
                                    (0, y_c0 - ay_i),
                                    (0, y_c0 + 1 - ay_i),
                                    (0, y_c0 + 2 - ay_i),
                                    (sx_cap, y_mid - ay_i),
                                    (sx_cap, 0),
                                ]
                                cap_offsets = list(dict.fromkeys(cap_offsets))
                            elif len(comp) >= 10 and 0 < ax_i < w_x - 1:
                                cap_offsets = [(-1, 0), (0, 0), (1, 0)]
                            elif len(comp) >= 10 and 0 < ay_i < w_y - 1:
                                cap_offsets = [(0, -1), (0, 0), (0, 1)]
                            elif len(comp) >= 8 and sx_cap != 0 and sy_cap != 0 and 0 <= ax_i + 2 * sx_cap < w_x and 0 <= ay_i + sy_cap < w_y:
                                cap_offsets = [(0, 0), (0, sy_cap), (sx_cap, sy_cap), (2 * sx_cap, sy_cap)]
                            elif len(comp) >= 8 and sx_cap != 0 and sy_cap != 0 and 0 <= ax_i + sx_cap < w_x and 0 <= ay_i + 2 * sy_cap < w_y:
                                cap_offsets = [(0, 0), (sx_cap, 0), (sx_cap, sy_cap), (sx_cap, 2 * sy_cap)]
                            else:
                                cap_offsets = [(0, 0), (sx_cap, 0) if sx_cap != 0 else (0, sy_cap)]
                            for dcx, dcy in cap_offsets:
                                cx_h, cy_h = ax_i + dcx, ay_i + dcy
                                if 0 <= cx_h < w_x and 0 <= cy_h < w_y:
                                    core_occ[cx_h, cy_h, p - 1] = True
                                    pier_protected[cx_h, cy_h, p - 1] = True
                                    if (cx_h, cy_h) in comp:
                                        pier_protected[cx_h, cy_h, p] = True
                                        if (cx_h, cy_h) not in supported_pts:
                                            supported_pts.append((cx_h, cy_h))
                            placed_thin_pedestal = True
                            eff_pier_rad = 10

    # Protect pier-top lintel anchors within Manhattan distance <= 2 of genuine unsupported bridge spans
    # so arches, doorways, tunnels, and cantilevers NEVER have their supporting pier shoulders carved!
    for p_br in range(1, n_plates_z):
        if p_br in jumper_bridge_layers or not np.any(core_occ[:, :, p_br]):
            continue
        p_bel = p_br - 1
        while p_bel > 0 and p_bel in jumper_bridge_layers:
            p_bel -= 1
        if layer_phase[p_br] != layer_phase[p_bel]:
            continue
        unsup_xy = core_occ[:, :, p_br] & (~core_occ[:, :, p_bel])
        n_unsup = int(np.count_nonzero(unsup_xy))
        if n_unsup == 0 or (not forbidden_support_boxes and not is_legged_frame and n_unsup < 3):
            continue
        uxs, uys = np.nonzero(unsup_xy)
        for ux_i, uy_i in zip(uxs.tolist(), uys.tolist()):
            for ax_i in range(max(0, ux_i - 2), min(w_x, ux_i + 3)):
                for ay_i in range(max(0, uy_i - 2), min(w_y, uy_i + 3)):
                    md_a = abs(ax_i - ux_i) + abs(ay_i - uy_i)
                    if md_a <= 2:
                        pier_protected[ax_i, ay_i, : p_br + 1] = True

    # Ensure ground seed exists even on ultra-thin wireframes (`e.g. 5 LDU blades`)
    if not np.any(core_occ[:, :, 0]):
        ph0 = layer_phase[0]
        cg0 = phase_grids[ph0][:, :, 0]
        best_seed_val = 0
        best_seed_xy = (0, 0)
        for ix in range(w_x):
            for iy in range(w_y):
                wb0 = LDUBox(ix * 20 + ph0[0], iy * 20 + ph0[1], 0, (ix + 1) * 20 + ph0[0], (iy + 1) * 20 + ph0[1], 8)
                if not _box_hits_forbidden(wb0, forbidden_support_boxes) and int(cg0[ix, iy]) > best_seed_val:
                    best_seed_val = int(cg0[ix, iy])
                    best_seed_xy = (ix, iy)
        if best_seed_val > 0:
            seeded = False
            for ix in range(w_x):
                for iy in range(w_y):
                    wb0 = LDUBox(ix * 20 + ph0[0], iy * 20 + ph0[1], 0, (ix + 1) * 20 + ph0[0], (iy + 1) * 20 + ph0[1], 8)
                    if not _box_hits_forbidden(wb0, forbidden_support_boxes) and int(cg0[ix, iy]) >= 14:
                        core_occ[ix, iy, 0] = True
                        seeded = True
            if not seeded:
                core_occ[best_seed_xy[0], best_seed_xy[1], 0] = True

    core_occ_pre_sculpt = core_occ.copy()

    # Exact `occ6` marginal sub-voxel terminal-plate prune (top-down): prune any topmost `core_occ[ix, iy, p]`
    # sitting on `core_occ[ix, iy, p - 1]` whose marginal `5x5x6 LDU` slice (`above ceil(8*p / 6)`) adds more
    # empty overshoot than mesh intersection (`in_marg < 0.45 * tot_marg` on thick meshes, `< 0.35` on thin frames).
    for _prune_pass in range(2):
        if is_thin_regime:
            marg_ratio = 0.20 if has_thin_panel_kit else 0.35
        else:
            iou_cur_pr = _est_core_occ6_iou(core_occ, layer_phase, jumper_bridge_layers)
            marg_ratio = max(0.455, iou_cur_pr / (1.0 + iou_cur_pr))
        for p_pr in range(n_plates_z - 1, 0, -1):
            if p_pr in jumper_bridge_layers or (p_pr - 1) in jumper_bridge_layers:
                continue
            ph_pr = layer_phase[p_pr]
            if layer_phase[p_pr - 1] != ph_pr:
                continue
            z6_prev = min(occ6.shape[2], int(math.ceil((8 * p_pr) / 6.0)))
            z6_cur = min(occ6.shape[2], int(math.ceil((8 * (p_pr + 1)) / 6.0)))
            if z6_cur <= z6_prev:
                continue
            for ix in range(w_x):
                for iy in range(w_y):
                    if not core_occ[ix, iy, p_pr] or not core_occ[ix, iy, p_pr - 1]:
                        continue
                    if pier_protected[ix, iy, p_pr] or inv_reserved_occ[ix, iy, p_pr]:
                        continue
                    if (p_pr + 1) < n_plates_z and (
                        core_occ[ix, iy, p_pr + 1 :].any() or inv_reserved_occ[ix, iy, p_pr + 1 :].any()
                    ):
                        continue
                    x6_p = (ix * 20 + ph_pr[0]) // 5
                    y6_p = (iy * 20 + ph_pr[1]) // 5
                    blk_m = occ6[x6_p : x6_p + 4, y6_p : y6_p + 4, z6_prev:z6_cur]
                    if blk_m.size == 0:
                        continue
                    in_m_cnt = int(np.count_nonzero(blk_m))
                    if float(in_m_cnt) < marg_ratio * float(blk_m.size):
                        core_occ[ix, iy, p_pr] = False

    snot_spine_protected = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
    snot_channel_carved = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
    has_sculptural_kit = any(
        (CATALOG_RETRIEVAL_INDEX.get_entry(pid) is not None and CATALOG_RETRIEVAL_INDEX.get_entry(pid).taxonomy_class in SCULPTURAL_TAXONOMY_CLASSES)
        for pid in kit_ids
    )
    if has_sculptural_kit and not is_thin_regime:
        # Build phase-aligned 3-plate brick course starts within each maximal non-jumper phase segment
        # so upper tiers above `15573` jumper bridges (`even when p_up0 % 3 != 0`!) get full perimeter slope erosion!
        phase_course_starts: List[int] = []
        seg_s = 0
        while seg_s < n_plates_z:
            if seg_s in jumper_bridge_layers:
                seg_s += 1
                continue
            seg_ph = layer_phase[seg_s]
            seg_e = seg_s + 1
            while seg_e < n_plates_z and seg_e not in jumper_bridge_layers and layer_phase[seg_e] == seg_ph:
                seg_e += 1
            p_cur = seg_s
            while p_cur + 3 <= seg_e:
                phase_course_starts.append(p_cur)
                p_cur += 3
            seg_s = seg_e

        for p0 in reversed(phase_course_starts):
            if p0 == 0 and foundation_policy != FoundationPolicy.CLUTCHED_BASEPLATE:
                continue
            p_top = p0 + 2
            if (p_top + 1) in jumper_bridge_layers:
                continue
            ph_c = layer_phase[p0]
            cg_c = phase_grids[ph_c]
            for ix in range(w_x):
                for iy in range(w_y):
                    if np.any(pier_protected[ix, iy, p0 : p_top + 1]) or np.any(inv_reserved_occ[ix, iy, p0 : p_top + 1]):
                        continue
                    if (p_top + 1) < n_plates_z and (
                        np.any(core_occ[ix, iy, p_top + 1 :]) or np.any(cg_c[ix, iy, p_top + 1 :] >= 16)
                    ):
                        continue
                    if not np.any(core_occ[ix, iy, p0 : p_top + 1]):
                        continue
                    v0 = int(cg_c[ix, iy, p0])
                    v1 = int(cg_c[ix, iy, p0 + 1])
                    v2 = int(cg_c[ix, iy, p0 + 2])
                    is_perim = any(
                        not (0 <= ix + dx_p < w_x and 0 <= iy + dy_p < w_y and core_occ[ix + dx_p, iy + dy_p, p0])
                        for dx_p, dy_p in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
                    if is_perim and v0 <= 22 and v1 <= 14 and v2 <= 8:
                        core_occ[ix, iy, p0 : p_top + 1] = False

        # Lateral half-stud stepped-terrace carving (top-down across phase-aligned courses):
        # carve outer perimeter cells whose outer `10 LDU` half is nearly empty (`outer_p0 <= 5` and `outer_cnt <= 16`)
        # while their inner `10 LDU` half is present (`>= 26 / 96`), exposing outward SNOT spines for `8 LDU` tiles!
        if has_snot_core:
            for p0_l in reversed(phase_course_starts):
                if p0_l == 0 and foundation_policy != FoundationPolicy.CLUTCHED_BASEPLATE:
                    continue
                if (p0_l + 3) in jumper_bridge_layers:
                    continue
                ph_l = layer_phase[p0_l]
                to_carve_lat: Dict[Tuple[int, int], Tuple[int, int, bool]] = {}
                for ix in range(w_x):
                    for iy in range(w_y):
                        if not np.any(core_occ[ix, iy, p0_l : p0_l + 3]):
                            continue
                        if np.any(pier_protected[ix, iy, p0_l : p0_l + 3]) or np.any(
                            inv_reserved_occ[ix, iy, p0_l : p0_l + 3]
                        ):
                            continue
                        if (p0_l + 3) < n_plates_z and np.any(core_occ[ix, iy, p0_l + 3 :]):
                            continue
                        sx0_l = (ix * 20 + ph_l[0]) // 5
                        sy0_l = (iy * 20 + ph_l[1]) // 5
                        blk_l = occ2[sx0_l : sx0_l + 4, sy0_l : sy0_l + 4, 4 * p0_l : 4 * (p0_l + 3)]
                        for dx_o, dy_o in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ox_i, oy_i = ix + dx_o, iy + dy_o
                            in_x, in_y = ix - dx_o, iy - dy_o
                            if 0 <= ox_i < w_x and 0 <= oy_i < w_y and np.any(core_occ[ox_i, oy_i, p0_l : p0_l + 3]):
                                continue
                            if not (0 <= in_x < w_x and 0 <= in_y < w_y and np.all(core_occ[in_x, in_y, p0_l : p0_l + 3])):
                                continue
                            if (in_x, in_y) in to_carve_lat:
                                continue
                            in2_x, in2_y = ix - 2 * dx_o, iy - 2 * dy_o
                            has_in2 = (
                                0 <= in2_x < w_x
                                and 0 <= in2_y < w_y
                                and bool(np.all(core_occ[in2_x, in2_y, p0_l : p0_l + 3]))
                                and (in2_x, in2_y) not in to_carve_lat
                            )
                            if not has_in2:
                                perp_dx, perp_dy = abs(dy_o), abs(dx_o)
                                has_perp_spine = any(
                                    0 <= in_x + sgn * perp_dx < w_x
                                    and 0 <= in_y + sgn * perp_dy < w_y
                                    and bool(np.all(core_occ[in_x + sgn * perp_dx, in_y + sgn * perp_dy, p0_l : p0_l + 3]))
                                    and (in_x + sgn * perp_dx, in_y + sgn * perp_dy) not in to_carve_lat
                                    for sgn in (-1, 1)
                                )
                                if not has_perp_spine:
                                    continue
                            if dx_o == 1:
                                outer_p0 = int(np.count_nonzero(blk_l[2:, :, :4]))
                                outer_cnt = int(np.count_nonzero(blk_l[2:, :, :]))
                                inner_cnt = int(np.count_nonzero(blk_l[:2, :, :]))
                            elif dx_o == -1:
                                outer_p0 = int(np.count_nonzero(blk_l[:2, :, :4]))
                                outer_cnt = int(np.count_nonzero(blk_l[:2, :, :]))
                                inner_cnt = int(np.count_nonzero(blk_l[2:, :, :]))
                            elif dy_o == 1:
                                outer_p0 = int(np.count_nonzero(blk_l[:, 2:, :4]))
                                outer_cnt = int(np.count_nonzero(blk_l[:, 2:, :]))
                                inner_cnt = int(np.count_nonzero(blk_l[:, :2, :]))
                            else:
                                outer_p0 = int(np.count_nonzero(blk_l[:, :2, :4]))
                                outer_cnt = int(np.count_nonzero(blk_l[:, :2, :]))
                                inner_cnt = int(np.count_nonzero(blk_l[:, 2:, :]))
                            if outer_p0 <= 6 and outer_cnt <= 18 and inner_cnt >= 28:
                                to_carve_lat[(ix, iy)] = (
                                    in_x,
                                    in_y,
                                    bool(outer_cnt <= 14 and inner_cnt >= 52),
                                )
                                break
                for (cx_l, cy_l), (sp_x, sp_y, is_vert_hs) in to_carve_lat.items():
                    core_occ[cx_l, cy_l, p0_l : p0_l + 3] = False
                    snot_channel_carved[cx_l, cy_l, p0_l : p0_l + 3] = True
                    if is_vert_hs and not (
                        (p0_l + 3) >= n_plates_z - 1
                        and int(phase_grids[ph_l][sp_x, sp_y, p0_l + 2]) < 44
                    ):
                        snot_spine_protected[sp_x, sp_y, p0_l : p0_l + 3] = True

    brick_templates = _rect_templates(kit_ids, 24)
    plate_templates = _rect_templates(kit_ids, 8)
    tile_templates = _rect_templates(kit_ids, 8, studless_tiles=True)
    if not plate_templates:
        plate_templates = [("3024", 1, 1)]
    if not brick_templates:
        brick_templates = [("3005", 1, 1)]
    if not tile_templates and inventory_counts is None:
        tile_templates = [
            ("87079", 2, 4),
            ("26603", 2, 3),
            ("3068b", 2, 2),
            ("2431", 1, 4),
            ("63864", 1, 3),
            ("3069b", 1, 2),
            ("3070b", 1, 1),
        ]
        if license_filter == "cc_by_4_0_only":
            tile_templates = [
                t
                for t in tile_templates
                if t[0] in EXTENDED_BRICK_CATALOG
                or (
                    CATALOG_RETRIEVAL_INDEX.get_entry(t[0]) is not None
                    and CATALOG_RETRIEVAL_INDEX.get_entry(t[0]).license_tier == "cc_by_4_0"  # type: ignore[union-attr]
                )
            ]

    planned_by_z: Dict[int, List[Tuple[str, LDUFrame]]] = {
        z_k: list(v_k) for z_k, v_k in planned_inv_by_z.items()
    }

    # Plan `15573` jumper transition layers FIRST so `jumper_top_stud_xy[p]` records the exact
    # shifted cells `(ux, uy)` at `p + 1` resting directly on a `15573` top stud!
    jumper_top_stud_xy: Dict[int, np.ndarray] = {}
    jumper_used_low: Dict[int, np.ndarray] = {}
    jumper_low_occ: Dict[int, np.ndarray] = {}
    planned_jb_y_targets: Dict[int, np.ndarray] = {}

    def _disjoint_target_indices(
        u_line: np.ndarray, low_line: np.ndarray, ph_to_ax: int, ph_from_ax: int, alt_parity: int = 0
    ) -> List[int]:
        """Select `100%` disjoint `15573` upper-tier stud indices covering both perimeter endpoints."""
        occ_u = np.nonzero(u_line)[0].tolist()
        if not occ_u:
            return []
        runs: List[Tuple[int, int]] = []
        r_s = occ_u[0]
        r_e = occ_u[0]
        for u_v in occ_u[1:]:
            if u_v == r_e + 1:
                r_e = u_v
            else:
                runs.append((r_s, r_e))
                r_s = u_v
                r_e = u_v
        runs.append((r_s, r_e))

        def _pair_supported(u_idx: int) -> bool:
            w_c = u_idx * 20 + ph_to_ax + 10
            i0 = (w_c - 10 - ph_from_ax) // 20
            if not (0 <= i0 and (i0 + 1) < low_line.shape[0]):
                return False
            return bool(low_line[i0] or low_line[i0 + 1]) if is_thin_regime else bool(low_line[i0] and low_line[i0 + 1])

        chosen: List[int] = []
        for u0, u1 in runs:
            span_w = u1 - u0 + 1
            if span_w % 2 == 1:
                if (
                    span_w == 3
                    and (alt_parity % 3 == 1)
                    and _pair_supported(u0 + 1)
                ):
                    cands_r = [u0 + 1]
                else:
                    cands_r = list(range(u0, u1 + 1, 2))
            elif span_w >= 4 and _pair_supported(u0) and _pair_supported(u1):
                cands_r = list(range(u0, u1 - 2, 2)) + [u1]
            elif span_w == 2 and _pair_supported(u0) and _pair_supported(u1):
                cands_r = [u1] if (alt_parity % 2 == 1) else [u0]
            elif _pair_supported(u0):
                cands_r = list(range(u0, u1 + 1, 2))
            else:
                cands_r = list(range(u0 + 1, u1 + 1, 2))
            for u_c in cands_r:
                if _pair_supported(u_c):
                    chosen.append(u_c)
        return chosen

    for p in sorted(jumper_bridge_layers.keys()):
        z0 = p * 8
        ph_from, ph_to = jumper_bridge_layers[p]
        shift_x = ph_to[0] != ph_from[0]
        p_low = p - 1
        if p_low in jumper_top_stud_xy:
            low_occ = jumper_top_stud_xy[p_low]
        elif p_low >= 0:
            low_occ = core_occ[:, :, p_low]
        elif foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE:
            low_occ = np.ones((w_x, w_y), dtype=bool)
        else:
            low_occ = np.zeros((w_x, w_y), dtype=bool)
        p_up_start = p + 1
        while p_up_start < n_plates_z and p_up_start in jumper_bridge_layers:
            p_up_start += 1
        ph_final = layer_phase[p_up_start] if p_up_start < n_plates_z else ph_to
        raw_up = np.zeros((w_x, w_y), dtype=bool)
        for q_u in range(p_up_start, min(n_plates_z, p_up_start + 3)):
            if layer_phase[q_u] == ph_final:
                raw_up |= core_occ[:, :, q_u]

        cg_jb_low = phase_grids[ph_from][:, :, p]
        if ph_final == ph_to:
            if p in planned_jb_y_targets:
                up_occ_union = planned_jb_y_targets[p]
            elif not is_thin_regime:
                up_occ_union = raw_up.copy()
                if shift_x:
                    if w_y >= 6:
                        if int(np.max(cg_jb_low[:, 0])) < 54 and up_occ_union[:, 1:].any():
                            up_occ_union[:, 0] = False
                        if int(np.max(cg_jb_low[:, w_y - 1])) < 54 and up_occ_union[:, : w_y - 1].any():
                            up_occ_union[:, w_y - 1] = False
                    occ_ux1 = np.nonzero(up_occ_union.any(axis=1))[0].tolist()
                    if len(occ_ux1) >= 5:
                        ix_0 = ((occ_ux1[0] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                        ix_1 = ((occ_ux1[-1] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                        if ix_0 == 0 and int(np.max(cg_jb_low[0, :])) < 56 and up_occ_union[occ_ux1[0] + 1 :, :].any():
                            up_occ_union[occ_ux1[0], :] = False
                        if (ix_1 + 1) == (w_x - 1) and int(np.max(cg_jb_low[w_x - 1, :])) < 56 and up_occ_union[: occ_ux1[-1], :].any():
                            up_occ_union[occ_ux1[-1], :] = False
                else:
                    if w_x >= 6:
                        if int(np.max(cg_jb_low[0, :])) < 54 and up_occ_union[1:, :].any():
                            up_occ_union[0, :] = False
                        if int(np.max(cg_jb_low[w_x - 1, :])) < 54 and up_occ_union[: w_x - 1, :].any():
                            up_occ_union[w_x - 1, :] = False
                    occ_uy1 = np.nonzero(up_occ_union.any(axis=0))[0].tolist()
                    if len(occ_uy1) >= 5:
                        iy_0 = ((occ_uy1[0] * 20 + ph_to[1] + 10) - 10 - ph_from[1]) // 20
                        iy_1 = ((occ_uy1[-1] * 20 + ph_to[1] + 10) - 10 - ph_from[1]) // 20
                        if iy_0 == 0 and int(np.max(cg_jb_low[:, 0])) < 56 and up_occ_union[:, occ_uy1[0] + 1 :].any():
                            up_occ_union[:, occ_uy1[0]] = False
                        if (iy_1 + 1) == (w_y - 1) and int(np.max(cg_jb_low[:, w_y - 1])) < 56 and up_occ_union[:, : occ_uy1[-1]].any():
                            up_occ_union[:, occ_uy1[-1]] = False
            else:
                up_occ_union = raw_up
        elif shift_x and (p + 1) in jumper_bridge_layers:
            # Top-down 2-layer `(X-shift at p, Y-shift at p + 1)` jumper bridge planning:
            # 1) Trim tapered outer X-columns (`ix == 0`, `ix == w_x - 1`), then select disjoint X-columns `u_cols_x`.
            # 2) On each `k_col, ux_c in enumerate(u_cols_x)`, select parity-alternated interior Y-targets `uy_t` at `p + 1`.
            # 3) Populate `up_occ_union` at `p` with ONLY the exact intermediate `(ux_c, iy_m), (ux_c, iy_m + 1)` cells!
            low_pair_ok_x = np.zeros(w_x, dtype=bool)
            for ix_chk in range(w_x - 1):
                if bool((low_occ[ix_chk, :] & low_occ[ix_chk + 1, :]).any()):
                    low_pair_ok_x[ix_chk] = True
                    low_pair_ok_x[ix_chk + 1] = True
            raw_x_line = raw_up.any(axis=1).copy()
            occ_ux = np.nonzero(raw_x_line)[0].tolist()
            if len(occ_ux) >= 3:
                trim_x = raw_x_line.copy()
                ix_x0 = ((occ_ux[0] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                ix_x1 = ((occ_ux[-1] * 20 + ph_to[0] + 10) - 10 - ph_from[0]) // 20
                if ix_x0 == 0 and int(np.max(cg_jb_low[0, :])) < 58:
                    trim_x[occ_ux[0]] = False
                if (ix_x1 + 1) == (w_x - 1) and int(np.max(cg_jb_low[w_x - 1, :])) < 58:
                    trim_x[occ_ux[-1]] = False
                if trim_x.any() and _disjoint_target_indices(trim_x, low_pair_ok_x, ph_to[0], ph_from[0], alt_parity=0):
                    raw_x_line = trim_x
            u_cols_x = _disjoint_target_indices(
                raw_x_line, low_pair_ok_x, ph_to[0], ph_from[0], alt_parity=0
            )
            y_target_mask = np.zeros((w_x, w_y), dtype=bool)
            up_occ_union = np.zeros((w_x, w_y), dtype=bool)
            for k_col, ux_c in enumerate(u_cols_x):
                w_cx_c = ux_c * 20 + ph_to[0] + 10
                ix_c = (w_cx_c - 10 - ph_from[0]) // 20
                if ix_c < 0 or ix_c + 1 >= w_x:
                    continue
                low_y_for_col = low_occ[ix_c, :] & low_occ[ix_c + 1, :]
                u_line_y = raw_up[ux_c, :].copy()
                occ_uy = np.nonzero(u_line_y)[0].tolist()
                if len(occ_uy) >= 4:
                    u0_y, u1_y = occ_uy[0], occ_uy[-1]
                    iym_0 = (u0_y * 20 + ph_final[1] - ph_to[1]) // 20
                    iym_1 = (u1_y * 20 + ph_final[1] - ph_to[1]) // 20
                    # When outer Tier-1 Y-walls (`iy == 0` or `iy == w_y - 1`) taper (`cg < 54`), keep jumper
                    # bridges strictly interior along Y so ALL 4 Tier-1 perimeter walls stay free for slopes!
                    trimmed_y = u_line_y.copy()
                    if iym_0 == 0 and max(int(cg_jb_low[ix_c, 0]), int(cg_jb_low[ix_c + 1, 0])) < 54:
                        trimmed_y[u0_y] = False
                    if (iym_1 + 1) == (w_y - 1) and max(
                        int(cg_jb_low[ix_c, w_y - 1]), int(cg_jb_low[ix_c + 1, w_y - 1])
                    ) < 54:
                        trimmed_y[u1_y] = False
                    if trimmed_y.any():
                        cand_trim = _disjoint_target_indices(
                            trimmed_y, low_y_for_col, ph_final[1], ph_to[1], alt_parity=k_col
                        )
                        if cand_trim:
                            u_line_y = trimmed_y
                uy_list = _disjoint_target_indices(
                    u_line_y, low_y_for_col, ph_final[1], ph_to[1], alt_parity=k_col
                )
                for uy_c in uy_list:
                    iy_m = (uy_c * 20 + ph_final[1] - ph_to[1]) // 20
                    if 0 <= iy_m and (iy_m + 1) < w_y and low_y_for_col[iy_m] and low_y_for_col[iy_m + 1]:
                        y_target_mask[ux_c, uy_c] = True
                        up_occ_union[ux_c, iy_m] = True
                        up_occ_union[ux_c, iy_m + 1] = True
            planned_jb_y_targets[p + 1] = y_target_mask
        else:
            up_occ_union = np.zeros((w_x, w_y), dtype=bool)
            uxs_r, uys_r = np.nonzero(raw_up)
            for uxf, uyf in zip(uxs_r.tolist(), uys_r.tolist()):
                if shift_x:
                    iy_m = (uyf * 20 + ph_final[1] - ph_to[1]) // 20
                    for iyc in (iy_m, iy_m + 1):
                        if 0 <= iyc < w_y:
                            up_occ_union[uxf, iyc] = True
                else:
                    ix_m = (uxf * 20 + ph_final[0] - ph_to[0]) // 20
                    for ixc in (ix_m, ix_m + 1):
                        if 0 <= ixc < w_x:
                            up_occ_union[ixc, uyf] = True

        used_low = np.zeros((w_x, w_y), dtype=bool)
        stud_mask_up = np.zeros((w_x, w_y), dtype=bool)
        prim_j = get_brick_primitive("15573")

        for rid_j in C4_Z_ROT_IDS:
            jf = LDUFrame(rid_j, (0, 0, 0))
            apts = [jf.compose(pt.local_frame).pos for pt in prim_j.anti_stud_ports]
            dx_a = max(a[0] for a in apts) - min(a[0] for a in apts)
            dy_a = max(a[1] for a in apts) - min(a[1] for a in apts)
            if (shift_x and dx_a == 20 and dy_a == 0) or ((not shift_x) and dy_a == 20 and dx_a == 0):
                min_ax = min(a[0] for a in apts)
                min_ay = min(a[1] for a in apts)
                act_line_k = 0
                use_alt_p = ((p + 1) not in jumper_bridge_layers) and (p not in planned_jb_y_targets)
                if shift_x:
                    is_preplanned_2layer = (p + 1) in planned_jb_y_targets
                    for iy in range(w_y):
                        u_targets = (
                            np.nonzero(up_occ_union[:, iy])[0].tolist()
                            if is_preplanned_2layer
                            else _disjoint_target_indices(
                                up_occ_union[:, iy],
                                low_occ[:, iy],
                                ph_to[0],
                                ph_from[0],
                                alt_parity=(act_line_k if use_alt_p else 0),
                            )
                        )
                        if u_targets:
                            act_line_k += 1
                        for ux_t in u_targets:
                            w_cx = ux_t * 20 + ph_to[0] + 10
                            ix = (w_cx - 10 - ph_from[0]) // 20
                            nx2, ny2 = ix + 1, iy
                            if ix < 0 or nx2 >= w_x or used_low[ix, iy] or used_low[nx2, ny2]:
                                continue
                            if not (
                                (low_occ[ix, iy] or low_occ[nx2, ny2])
                                if is_thin_regime
                                else (low_occ[ix, iy] and low_occ[nx2, ny2])
                            ):
                                continue
                            if rem_inv is not None and rem_inv.get("15573", 0) <= 0:
                                continue
                            tx = ix * 20 + ph_from[0] + 10 - min_ax
                            ty = iy * 20 + ph_from[1] + 10 - min_ay
                            tz = z0 - apts[0][2]
                            pose_j = LDUFrame(rid_j, (tx, ty, tz))
                            wb_j = prim_j.hull_boxes[0].transform(pose_j)
                            if _box_hits_forbidden(wb_j, forbidden_support_boxes):
                                continue
                            planned_by_z.setdefault(z0, []).append(("15573", pose_j))
                            used_low[ix, iy] = True
                            used_low[nx2, ny2] = True
                            for pt_s in prim_j.ports:
                                if pt_s.polarity == PortPolarity.STUD:
                                    sp = pose_j.compose(pt_s.local_frame).pos
                                    ux_s = (sp[0] - ph_to[0]) // 20
                                    uy_s = (sp[1] - ph_to[1]) // 20
                                    if 0 <= ux_s < w_x and 0 <= uy_s < w_y:
                                        stud_mask_up[ux_s, uy_s] = True
                            if rem_inv is not None:
                                rem_inv["15573"] = rem_inv.get("15573", 0) - 1
                else:
                    is_preplanned_y = p in planned_jb_y_targets
                    for ix in range(w_x):
                        u_targets = (
                            np.nonzero(up_occ_union[ix, :])[0].tolist()
                            if is_preplanned_y
                            else _disjoint_target_indices(
                                up_occ_union[ix, :],
                                low_occ[ix, :],
                                ph_to[1],
                                ph_from[1],
                                alt_parity=(act_line_k if use_alt_p else 0),
                            )
                        )
                        if u_targets:
                            act_line_k += 1
                        for uy_t in u_targets:
                            w_cy = uy_t * 20 + ph_to[1] + 10
                            iy = (w_cy - 10 - ph_from[1]) // 20
                            nx2, ny2 = ix, iy + 1
                            if iy < 0 or ny2 >= w_y or used_low[ix, iy] or used_low[nx2, ny2]:
                                continue
                            if not (
                                (low_occ[ix, iy] or low_occ[nx2, ny2])
                                if is_thin_regime
                                else (low_occ[ix, iy] and low_occ[nx2, ny2])
                            ):
                                continue
                            if rem_inv is not None and rem_inv.get("15573", 0) <= 0:
                                continue
                            tx = ix * 20 + ph_from[0] + 10 - min_ax
                            ty = iy * 20 + ph_from[1] + 10 - min_ay
                            tz = z0 - apts[0][2]
                            pose_j = LDUFrame(rid_j, (tx, ty, tz))
                            wb_j = prim_j.hull_boxes[0].transform(pose_j)
                            if _box_hits_forbidden(wb_j, forbidden_support_boxes):
                                continue
                            planned_by_z.setdefault(z0, []).append(("15573", pose_j))
                            used_low[ix, iy] = True
                            used_low[nx2, ny2] = True
                            for pt_s in prim_j.ports:
                                if pt_s.polarity == PortPolarity.STUD:
                                    sp = pose_j.compose(pt_s.local_frame).pos
                                    ux_s = (sp[0] - ph_to[0]) // 20
                                    uy_s = (sp[1] - ph_to[1]) // 20
                                    if 0 <= ux_s < w_x and 0 <= uy_s < w_y:
                                        stud_mask_up[ux_s, uy_s] = True
                            if rem_inv is not None:
                                rem_inv["15573"] = rem_inv.get("15573", 0) - 1
                break

        jumper_top_stud_xy[p] = stud_mask_up
        jumper_used_low[p] = used_low.copy()
        jumper_low_occ[p] = low_occ.copy()

    def _support_mask_at_layer(p_layer: int) -> Optional[np.ndarray]:
        if p_layer <= 0:
            return None
        if (p_layer - 1) in jumper_top_stud_xy:
            return jumper_top_stud_xy[p_layer - 1]
        ph_t = layer_phase[p_layer]
        p_b = p_layer - 1
        while p_b > 0 and p_b in jumper_bridge_layers:
            p_b -= 1
        ph_b = layer_phase[p_b]
        if ph_b == ph_t:
            return core_occ[:, :, p_b]
        sup_m = np.zeros((w_x, w_y), dtype=bool)
        for ix in range(w_x):
            for iy in range(w_y):
                xb0 = ix * 20 + ph_t[0]
                yb0 = iy * 20 + ph_t[1]
                for bx in range(max(0, ix - 1), min(w_x, ix + 2)):
                    for by in range(max(0, iy - 1), min(w_y, iy + 2)):
                        if core_occ[bx, by, p_b]:
                            if (
                                min(xb0 + 20, bx * 20 + ph_b[0] + 20) > max(xb0, bx * 20 + ph_b[0])
                                and min(yb0 + 20, by * 20 + ph_b[1] + 20) > max(yb0, by * 20 + ph_b[1])
                            ):
                                sup_m[ix, iy] = True
                                break
                    if sup_m[ix, iy]:
                        break
        return sup_m

    # Reserve multi-height Open-Catalog slopes, curved caps, wedges/rounds & thin structural panels
    # directly replacing coarse rectangular `core_occ` cells wherever exact global `occ6` `IoU` improves!
    slope_reserved_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
    must_be_smooth_tile_occ = np.zeros((w_x, w_y, n_plates_z), dtype=bool)
    has_3070b_for_corner = ("3070b" in kit_set) and (rem_inv is None or rem_inv.get("3070b", 0) > 0)
    has_top_stud_at_p = core_occ.copy()
    for jb_p, jb_mask in jumper_top_stud_xy.items():
        has_top_stud_at_p[:, :, jb_p] = jb_mask

    # Compute exact initial `occ6` intersection & overshoot of `core_occ` so `top_stud_slope_molds`
    # optimizes true global `ΔIoU` (`never carving dense material out of sparse/thin meshes!`)
    core_occ6_est = np.zeros_like(occ6)
    for p_est in range(n_plates_z):
        if p_est in jumper_bridge_layers or not np.any(core_occ[:, :, p_est]):
            continue
        ph_e0 = layer_phase[p_est]
        z6_e0 = max(0, (p_est * 8) // 6)
        z6_e1 = min(occ6.shape[2], int(math.ceil(((p_est + 1) * 8) / 6.0)))
        for ix_e0 in range(w_x):
            for iy_e0 in range(w_y):
                if core_occ[ix_e0, iy_e0, p_est]:
                    x6_e0 = (ix_e0 * 20 + ph_e0[0]) // 5
                    y6_e0 = (iy_e0 * 20 + ph_e0[1]) // 5
                    core_occ6_est[x6_e0 : x6_e0 + 4, y6_e0 : y6_e0 + 4, z6_e0:z6_e1] = True
    mesh_sub_est = max(1, int(np.count_nonzero(occ6)))
    core_inter_est = int(np.count_nonzero(core_occ6_est & occ6))
    core_over_est = int(np.count_nonzero(core_occ6_est & (~occ6)))
    base_iou_est = float(core_inter_est) / float(max(1, mesh_sub_est + core_over_est))

    allowed_top_slope_classes = (
        ("PLANAR_SLOPE", "STEEP_SLOPE", "XY_WEDGE_ROUND", "CURVED_SLOPE", "THIN_PANEL")
        if is_thin_regime
        else ("PLANAR_SLOPE", "STEEP_SLOPE", "XY_WEDGE_ROUND", "CURVED_SLOPE")
    )
    top_stud_slope_molds = [
        pid
        for pid in kit_ids
        if (ent_ts := CATALOG_RETRIEVAL_INDEX.get_entry(pid)) is not None
        and ent_ts.taxonomy_class in allowed_top_slope_classes
        and 0.20 <= ent_ts.fill_ratio <= 0.94
        and ent_ts.height_ldu in (8, 16, 24, 32, 48, 72)
        and max(ent_ts.studs_x, ent_ts.studs_y) >= 1
        and not (
            is_thin_regime
            and ent_ts.taxonomy_class != "THIN_PANEL"
            and ent_ts.studs_x >= 2
            and ent_ts.studs_y >= 2
            and ent_ts.height_ldu >= 24
        )
        and (
            len(get_brick_primitive(pid).hull_boxes) >= 2
            or (
                ent_ts.taxonomy_class == "THIN_PANEL"
                and len(get_brick_primitive(pid).hull_boxes) == 1
                and min(
                    get_brick_primitive(pid).hull_boxes[0].xmax - get_brick_primitive(pid).hull_boxes[0].xmin,
                    get_brick_primitive(pid).hull_boxes[0].ymax - get_brick_primitive(pid).hull_boxes[0].ymin,
                ) <= 12
            )
        )
        and not all(
            (b.xmax - b.xmin) % 20 == 0 and (b.ymax - b.ymin) % 20 == 0 and (b.zmax - b.zmin) % 24 == 0
            for b in get_brick_primitive(pid).hull_boxes
        )
    ]
    if top_stud_slope_molds:
        all_sl_cands: List[
            Tuple[
                float,
                float,
                str,
                LDUFrame,
                int,
                int,
                int,
                int,
                int,
                int,
                Set[Tuple[int, int]],
                Set[Tuple[int, int]],
                Set[Tuple[int, int]],
            ]
        ] = []
        base_cg_min = 4 if is_thin_regime else 20
        def _canonical_s3_templates(pid_s3: str) -> Tuple[Any, ...]:
            """Keep 1 canonical anchor `(drx0 == 0, dry0 == 0)` per rotation so Step 3 avoids 5x redundant anti-stud scans."""
            raw_t = _fast_slope_grid_templates(pid_s3)
            seen_r: Set[int] = set()
            dedup: List[Any] = []
            for t_item in raw_t:
                if t_item[4] == 0 and t_item[6] == 0 and t_item[0] not in seen_r:
                    seen_r.add(t_item[0])
                    dedup.append(t_item)
            if len(dedup) < len({t_item[0] for t_item in raw_t}):
                for t_item in raw_t:
                    if t_item[0] not in seen_r:
                        seen_r.add(t_item[0])
                        dedup.append(t_item)
            return tuple(dedup)

        fp_groups_s3: Dict[Tuple[int, int, int, int, int], List[Tuple[str, bool, str, Any]]] = {}
        for pid_sm in top_stud_slope_molds:
            ent_sm = CATALOG_RETRIEVAL_INDEX.get_entry(pid_sm)
            cls_sm = ent_sm.taxonomy_class if ent_sm is not None else ""
            is_tp_sm = cls_sm == "THIN_PANEL"
            for t_sm in _canonical_s3_templates(pid_sm):
                fp_groups_s3.setdefault((t_sm[4], t_sm[5], t_sm[6], t_sm[7], t_sm[8]), []).append(
                    (pid_sm, is_tp_sm, cls_sm, t_sm)
                )
        fp_group_list_s3 = tuple((k_fp, tuple(v_fp), any(it[1] for it in v_fp)) for k_fp, v_fp in fp_groups_s3.items())
        uniform_phase_s3 = len(set(layer_phase)) <= 1
        has_jb_s3 = bool(jumper_bridge_layers)
        step3_hard_block = inv_reserved_occ | snot_spine_protected
        has_hard_block_s3 = bool(step3_hard_block.any())
        has_pier_s3 = bool(pier_protected.any())
        core_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in core_occ.tolist()]
        hb_zb = (
            [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in step3_hard_block.tolist()]
            if has_hard_block_s3
            else None
        )
        pier_zb = (
            [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in pier_protected.tolist()]
            if has_pier_s3
            else None
        )
        inv_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in inv_reserved_occ.tolist()]
        pil_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in pillar_added_occ.tolist()]
        snot_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in snot_spine_protected.tolist()]
        perim_offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2))
        for p_sl0 in range(n_plates_z):
            if p_sl0 in jumper_bridge_layers or (p_sl0 == 0 and foundation_policy != FoundationPolicy.CLUTCHED_BASEPLATE):
                continue
            ph_s = layer_phase[p_sl0]
            cg_p0_l = phase_grids[ph_s][:, :, p_sl0].tolist()
            sup_s = _support_mask_at_layer(p_sl0)
            sup_s_l = sup_s.tolist() if sup_s is not None else None
            z0_s = p_sl0 * 8
            r_zmod3 = p_sl0 % 3
            bit_p0 = 1 << p_sl0
            bit_pm1 = (1 << (p_sl0 - 1)) if p_sl0 >= 1 else 0
            bit_pm2 = (1 << (p_sl0 - 2)) if p_sl0 >= 2 else 0
            bit_roof_chk = 1 << min(n_plates_z - 1, p_sl0 + 3)
            seen_sl_poses: Set[Tuple[str, int, int, int]] = set()

            for ix_r in range(w_x):
                core_zb_ix = core_zb[ix_r]
                cg_ix = cg_p0_l[ix_r]
                sup_ix = sup_s_l[ix_r] if sup_s_l is not None else None
                for iy_r in range(w_y):
                    cell_in_core = (core_zb_ix[iy_r] & bit_p0) != 0
                    cg_val_r = cg_ix[iy_r]
                    cell_in_mesh_s3 = cg_val_r >= base_cg_min or (
                        not is_thin_regime
                        and cg_val_r >= 12
                        and (
                            (ix_r + 1 < w_x and (core_zb[ix_r + 1][iy_r] & bit_p0) != 0)
                            or (iy_r + 1 < w_y and (core_zb_ix[iy_r + 1] & bit_p0) != 0)
                        )
                    )
                    if not cell_in_core and not cell_in_mesh_s3 and not (is_thin_regime and has_thin_panel_kit):
                        continue
                    if sup_ix is not None and not sup_ix[iy_r]:
                        continue
                    # Candidate must touch or sit within 2 cells of course/roof boundary (`or be on thin regime`)
                    if not is_thin_regime:
                        is_near_perim = (not cell_in_core) or (p_sl0 + 3 >= n_plates_z) or ((core_zb_ix[iy_r] & bit_roof_chk) == 0)
                        if not is_near_perim:
                            for dx_p, dy_p in perim_offsets:
                                nx_p, ny_p = ix_r + dx_p, iy_r + dy_p
                                if nx_p < 0 or nx_p >= w_x or ny_p < 0 or ny_p >= w_y or (core_zb[nx_p][ny_p] & bit_p0) == 0:
                                    is_near_perim = True
                                    break
                        if not is_near_perim:
                            continue
                    px_r = ix_r * 20 + ph_s[0] + 10
                    py_r = iy_r * 20 + ph_s[1] + 10
                    for (drx0, drx1, dry0, dry1, h_pl_s), grp_items_s3, grp_has_tp in fp_group_list_s3:
                        if not cell_in_core and not cell_in_mesh_s3 and not grp_has_tp:
                            continue
                        rx0_s = ix_r + drx0
                        rx1_s = ix_r + drx1
                        ry0_s = iy_r + dry0
                        ry1_s = iy_r + dry1
                        if rx0_s < 0 or ry0_s < 0 or rx1_s > w_x or ry1_s > w_y:
                            continue
                        p_top_next = p_sl0 + h_pl_s
                        if p_top_next > n_plates_z:
                            continue
                        z_span_m = ((1 << h_pl_s) - 1) << p_sl0
                        if not uniform_phase_s3 or has_jb_s3:
                            ph_bad = False
                            for q_h in range(h_pl_s):
                                pq_h = p_sl0 + q_h
                                if layer_phase[pq_h] != ph_s or (
                                    pq_h in jumper_bridge_layers
                                    and bool(jumper_used_low[pq_h][rx0_s:rx1_s, ry0_s:ry1_s].any())
                                ):
                                    ph_bad = True
                                    break
                            if ph_bad:
                                continue
                        if hb_zb is not None:
                            hb_hit = False
                            for bx_h in range(rx0_s, rx1_s):
                                hb_row = hb_zb[bx_h]
                                for by_h in range(ry0_s, ry1_s):
                                    if hb_row[by_h] & z_span_m:
                                        hb_hit = True
                                        break
                                if hb_hit:
                                    break
                            if hb_hit:
                                continue
                        pier_hit_fp = False
                        if pier_zb is not None:
                            for bx_p in range(rx0_s, rx1_s):
                                p_row = pier_zb[bx_p]
                                for by_p in range(ry0_s, ry1_s):
                                    if p_row[by_p] & z_span_m:
                                        pier_hit_fp = True
                                        break
                                if pier_hit_fp:
                                    break
                            if pier_hit_fp and not (is_thin_regime and grp_has_tp and not forbidden_support_boxes):
                                continue
                        sx0_q = (rx0_s * 20 + ph_s[0]) // 5
                        sx1_q = (rx1_s * 20 + ph_s[0]) // 5
                        sy0_q = (ry0_s * 20 + ph_s[1]) // 5
                        sy1_q = (ry1_s * 20 + ph_s[1]) // 5
                        sz0_q = z0_s // 6
                        sz1_q = int(math.ceil((p_top_next * 8) / 6.0))
                        tgt_blk = occ6[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                        cached_co_full: Optional[Tuple[int, int]] = None

                        for pid_sm, is_thin_panel_mold, cls_sm, t_sm in grp_items_s3:
                            if not cell_in_core and not cell_in_mesh_s3 and not is_thin_panel_mold:
                                continue
                            if pier_hit_fp and not (is_thin_regime and is_thin_panel_mold and not forbidden_support_boxes):
                                continue
                            if rem_inv is not None and rem_inv.get(pid_sm, 0) <= 0:
                                continue
                            (
                                rid_sm,
                                dx_sm,
                                dy_sm,
                                dz_sm,
                                _,
                                _,
                                _,
                                _,
                                _,
                                rboxes_sm,
                                rel_tstuds,
                                rel_banti,
                                rel_stud_clash,
                                rel_occ_cells,
                                masks_z3,
                            ) = t_sm
                            wx_sm = px_r + dx_sm
                            wy_sm = py_r + dy_sm
                            skey = (pid_sm, rid_sm, wx_sm, wy_sm)
                            if skey in seen_sl_poses:
                                continue
                            seen_sl_poses.add(skey)
                            pos_sm = (wx_sm, wy_sm, z0_s + dz_sm)
                            banti_xy_s: Set[Tuple[int, int]] = {
                                (ix_r + dbx, iy_r + dby) for dbx, dby in rel_banti
                            }
                            clash_xy_s: Set[Tuple[int, int]] = {
                                (ix_r + dcx, iy_r + dcy) for dcx, dcy in rel_stud_clash
                            }
                            # Require full downward support on anti-stud cells (`or >= 50% on 15573 jumper bridges`);
                            # on chamfered corner cells in `clash_xy_s`, allow capping with `3070b` so zero stud clash occurs!
                            if sup_s_l is not None:
                                on_jb_s = (p_sl0 - 1) in jumper_top_stud_xy
                                if on_jb_s:
                                    sup_ok = (
                                        sum(1 for bx_a, by_a in banti_xy_s if sup_s_l[bx_a][by_a])
                                        >= max(1, (len(banti_xy_s) + 1) // 2)
                                    ) and not any(sup_s_l[bx_c][by_c] for bx_c, by_c in clash_xy_s)
                                else:
                                    sup_ok = True
                                    for bx_a in range(rx0_s, rx1_s):
                                        sup_row_a = sup_s_l[bx_a]
                                        for by_a in range(ry0_s, ry1_s):
                                            if (bx_a, by_a) in banti_xy_s:
                                                if not sup_row_a[by_a]:
                                                    sup_ok = False
                                                    break
                                            elif (bx_a, by_a) in clash_xy_s and p_sl0 > 0 and (core_zb[bx_a][by_a] & bit_pm1) != 0:
                                                can_smooth_cap = (
                                                    has_3070b_for_corner
                                                    and (inv_zb[bx_a][by_a] & bit_pm1) == 0
                                                    and (p_sl0 == 1 or (core_zb[bx_a][by_a] & bit_pm2) != 0)
                                                )
                                                if not can_smooth_cap:
                                                    sup_ok = False
                                                    break
                                        if not sup_ok:
                                            break
                                if not sup_ok:
                                    continue
                            top_stud_xy_s: Set[Tuple[int, int]] = {
                                (ix_r + dtx, iy_r + dty) for dtx, dty in rel_tstuds
                            }
                            # Require every anti-stud base cell at `p_sl0` to lie inside mesh/core (`never spear empty air`),
                            # except grounded `p_sl0 == 0` thin structural wall panels where >= 1 base stud anchors a leg/pier!
                            if (
                                is_thin_regime
                                and is_thin_panel_mold
                                and p_sl0 == 0
                                and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                            ):
                                base_ok = any(
                                    (core_zb[bx_a][by_a] & 1) != 0 or cg_p0_l[bx_a][by_a] >= base_cg_min
                                    for bx_a, by_a in banti_xy_s
                                )
                            else:
                                has_core_back_stud = False
                                if top_stud_xy_s:
                                    for tx_s, ty_s in top_stud_xy_s:
                                        if (core_zb[tx_s][ty_s] & bit_p0) != 0 and (pil_zb[tx_s][ty_s] & bit_p0) == 0:
                                            has_core_back_stud = True
                                            break
                                base_ok = True
                                for bx_a, by_a in banti_xy_s:
                                    in_co = (
                                        (core_zb[bx_a][by_a] & bit_p0) != 0
                                        if (is_thin_regime and is_thin_panel_mold)
                                        else ((core_zb[bx_a][by_a] & bit_p0) != 0 and (pil_zb[bx_a][by_a] & bit_p0) == 0)
                                    )
                                    min_cg_cell = (
                                        12
                                        if (has_core_back_stud and (bx_a, by_a) not in top_stud_xy_s)
                                        else base_cg_min
                                    )
                                    if not (in_co or cg_p0_l[bx_a][by_a] >= min_cg_cell):
                                        base_ok = False
                                        break
                            if not base_ok:
                                continue

                            sm_blk = masks_z3[r_zmod3]
                            if sm_blk.shape != tgt_blk.shape:
                                continue
                            # Prevent tall slopes (`>= 32 LDU` = `h_pl_s >= 4`) from spearing empty sky at their apex!
                            if not is_thin_regime and not is_thin_panel_mold and h_pl_s >= 4 and sm_blk.shape[2] >= 2:
                                top_sl_m = sm_blk[:, :, -2:]
                                top_sl_tot = int(np.count_nonzero(top_sl_m))
                                if top_sl_tot > 0 and int(np.count_nonzero(top_sl_m & tgt_blk[:, :, -2:])) * 2 < top_sl_tot:
                                    continue

                            # Precomputed XY grid cells touched by `sm_blk` (`preserves untouched inner cells on L-shaped/macaroni `3063a`!`)
                            occ_cells_s = tuple((ix_r + domx, iy_r + domy) for domx, domy in rel_occ_cells)
                            if not occ_cells_s:
                                continue
                            is_partial_xy_mold = len(occ_cells_s) < (rx1_s - rx0_s) * (ry1_s - ry0_s)

                            upper_blocked = False
                            sh_clear_list: List[Tuple[int, int, int]] = []
                            if p_top_next < n_plates_z:
                                bit_ptop = 1 << p_top_next
                                if p_top_next in jumper_bridge_layers:
                                    j_req = jumper_used_low.get(p_top_next)
                                    if j_req is not None:
                                        for ux_k, uy_k in occ_cells_s:
                                            if j_req[ux_k, uy_k] and (ux_k, uy_k) not in top_stud_xy_s:
                                                upper_blocked = True
                                                break
                                elif layer_phase[p_top_next] == ph_s:
                                    for ux_k, uy_k in occ_cells_s:
                                        if (ux_k, uy_k) in top_stud_xy_s:
                                            continue
                                        if (inv_zb[ux_k][uy_k] & bit_ptop) != 0:
                                            upper_blocked = True
                                            break
                                        if (core_zb[ux_k][uy_k] & bit_ptop) != 0:
                                            can_clear_lip = (
                                                not is_thin_regime
                                                and h_pl_s >= 2
                                                and (pier_zb is None or (pier_zb[ux_k][uy_k] & bit_ptop) == 0)
                                                and (snot_zb[ux_k][uy_k] & bit_ptop) == 0
                                                and (
                                                    (p_top_next + 1) >= n_plates_z
                                                    or (
                                                        (core_zb[ux_k][uy_k] >> (p_top_next + 1)) == 0
                                                        and (inv_zb[ux_k][uy_k] >> (p_top_next + 1)) == 0
                                                        and (
                                                            (p_top_next + 1) not in jumper_used_low
                                                            or not jumper_used_low[p_top_next + 1][ux_k, uy_k]
                                                        )
                                                    )
                                                )
                                            )
                                            if can_clear_lip:
                                                sh_clear_list.append((ux_k, uy_k, p_top_next))
                                            else:
                                                upper_blocked = True
                                                break
                                else:
                                    ph_u = layer_phase[p_top_next]
                                    for ux_k, uy_k in occ_cells_s:
                                        if (ux_k, uy_k) in top_stud_xy_s:
                                            continue
                                        xb0_k = ux_k * 20 + ph_s[0]
                                        yb0_k = uy_k * 20 + ph_s[1]
                                        for bx_u in range(max(0, ux_k - 1), min(w_x, ux_k + 2)):
                                            for by_u in range(max(0, uy_k - 1), min(w_y, uy_k + 2)):
                                                if (core_zb[bx_u][by_u] & bit_ptop) != 0:
                                                    if (
                                                        min(xb0_k + 20, bx_u * 20 + ph_u[0] + 20)
                                                        > max(xb0_k, bx_u * 20 + ph_u[0])
                                                        and min(yb0_k + 20, by_u * 20 + ph_u[1] + 20)
                                                        > max(yb0_k, by_u * 20 + ph_u[1])
                                                    ):
                                                        upper_blocked = True
                                                        break
                                            if upper_blocked:
                                                break
                                        if upper_blocked:
                                            break
                            if upper_blocked:
                                continue

                            if sz1_q > gz6_nonempty_limit:
                                sky_sl = sm_blk[:, :, max(0, gz6_nonempty_limit - sz0_q) :]
                                if int(np.count_nonzero(sky_sl)) > max(
                                    4, int(0.08 * int(np.count_nonzero(sm_blk & tgt_blk)))
                                ):
                                    continue
                            # Prevent multi-cell slopes (`e.g. 3040a`) from rotating 90° parallel to an inset cliff
                            # and shoving their full-height solid back-stud column (`top_stud_xy_s`) into empty sky!
                            if not is_thin_regime and not is_thin_panel_mold and top_stud_xy_s:
                                p_back_top = p_sl0 + h_pl_s - 1
                                if any(
                                    int(phase_grids[ph_s][tx_b, ty_b, p_back_top]) < 46
                                    for tx_b, ty_b in top_stud_xy_s
                                ):
                                    continue
                            # Never let a studless (`len(rel_tstuds) == 0`) slope cap substantial uncovered mesh above it
                            if (
                                len(rel_tstuds) == 0
                                and sz1_q < occ6.shape[2]
                                and not (is_thin_panel_mold and (sz1_q >= gz6_nonempty_limit - 1 or h_pl_s >= 3))
                            ):
                                uz2_q = min(occ6.shape[2], sz1_q + 4)
                                above_mesh_q = int(np.count_nonzero(occ6[sx0_q:sx1_q, sy0_q:sy1_q, sz1_q:uz2_q]))
                                min_uncov_cap = (
                                    3 if is_thin_regime else 10 * max(1, len(occ_cells_s))
                                )
                                if above_mesh_q >= min_uncov_cap:
                                    continue

                            sm_in = int(np.count_nonzero(sm_blk & tgt_blk))
                            sm_tot = int(np.count_nonzero(sm_blk))
                            if is_partial_xy_mold:
                                co_in = 0
                                co_tot = 0
                                for ux_oc, uy_oc in occ_cells_s:
                                    cx_oc = (ux_oc * 20 + ph_s[0]) // 5
                                    cy_oc = (uy_oc * 20 + ph_s[1]) // 5
                                    c_b_oc = core_occ6_est[cx_oc : cx_oc + 4, cy_oc : cy_oc + 4, sz0_q:sz1_q]
                                    t_b_oc = occ6[cx_oc : cx_oc + 4, cy_oc : cy_oc + 4, sz0_q:sz1_q]
                                    co_in += int(np.count_nonzero(c_b_oc & t_b_oc))
                                    co_tot += int(np.count_nonzero(c_b_oc))
                            else:
                                if cached_co_full is None:
                                    co_blk = core_occ6_est[sx0_q:sx1_q, sy0_q:sy1_q, sz0_q:sz1_q]
                                    cached_co_full = (int(np.count_nonzero(co_blk & tgt_blk)), int(np.count_nonzero(co_blk)))
                                co_in, co_tot = cached_co_full
                            if sh_clear_list:
                                sz1_sh = min(occ6.shape[2], int(math.ceil(((p_top_next + 1) * 8) / 6.0)))
                                if sz1_sh > sz1_q:
                                    for ux_cl, uy_cl, _ in sh_clear_list:
                                        cx6_cl = (ux_cl * 20 + ph_s[0]) // 5
                                        cy6_cl = (uy_cl * 20 + ph_s[1]) // 5
                                        c_sub_cl = core_occ6_est[cx6_cl : cx6_cl + 4, cy6_cl : cy6_cl + 4, sz1_q:sz1_sh]
                                        t_sub_cl = occ6[cx6_cl : cx6_cl + 4, cy6_cl : cy6_cl + 4, sz1_q:sz1_sh]
                                        co_in += int(np.count_nonzero(c_sub_cl & t_sub_cl))
                                        co_tot += int(np.count_nonzero(c_sub_cl))
                            if is_thin_regime and not is_thin_panel_mold:
                                if h_pl_s >= 4 and sm_tot >= 40:
                                    continue
                                if sm_tot >= 28 and sm_in < int(0.55 * (sm_tot - sm_in)):
                                    continue
                            d_in_sw = sm_in - co_in
                            d_ov_sw = (sm_tot - sm_in) - (co_tot - co_in)
                            new_iou_sw = float(max(0, core_inter_est + d_in_sw)) / float(
                                max(1, mesh_sub_est + core_over_est + d_ov_sw)
                            )
                            val_diff = (new_iou_sw - base_iou_est) * float(mesh_sub_est + core_over_est)
                            min_vd_req = (
                                (0.75 if is_thin_panel_mold else 1.2)
                                if is_thin_regime
                                else (0.70 if (len(rel_tstuds) == 0 and h_pl_s == 2 and p_top_next in jumper_bridge_layers) else 1.8)
                            )
                            is_curved_step_tie = (
                                cls_sm in ("CURVED_SLOPE", "XY_WEDGE_ROUND")
                                and not sh_clear_list
                                and sm_in >= 12
                                and sm_in >= co_in
                                and (sm_tot - sm_in) <= (co_tot - co_in)
                                and (d_in_sw > 0 or d_ov_sw < 0)
                            )
                            if sm_in >= 4 and (val_diff >= min_vd_req or is_curved_step_tie):
                                if forbidden_support_boxes and any(
                                    _box_hits_forbidden(
                                        LDUBox(b[0] + px_r, b[1] + py_r, b[2] + z0_s, b[3] + px_r, b[4] + py_r, b[5] + z0_s),
                                        forbidden_support_boxes,
                                    )
                                    for b in rboxes_sm
                                ):
                                    continue
                                area_xy_s = max(1, len(occ_cells_s))
                                taper_area_s = (
                                    float(area_xy_s)
                                    if len(top_stud_xy_s) == 0
                                    else float(max(1, len(occ_cells_s) - len(top_stud_xy_s)))
                                )
                                eff_vol_s = taper_area_s * (
                                    (float(h_pl_s) / 2.5)
                                    if (h_pl_s >= 4 and not is_thin_panel_mold and not forbidden_support_boxes)
                                    else 1.0
                                )
                                mold_h_s = sum(ord(ch) * (i_ch + 3) for i_ch, ch in enumerate(pid_sm))
                                mesh_h_s = w_x * 17 + w_y * 31 + n_plates_z * 13 + (mesh_sub_est % 29) * 7 + p_sl0 * 5
                                m_rot_tie = 0.014 * float(
                                    ((mesh_h_s ^ (mold_h_s * 19)) + mold_h_s * (1 + (mesh_h_s % 7))) % 11
                                )
                                wide_scale_s = (4.0 / float(area_xy_s)) if area_xy_s > 4 else 1.0
                                vd_norm_s = val_diff * wide_scale_s
                                ev_norm_s = max(0.5, eff_vol_s * wide_scale_s)
                                sc_sm = (
                                    (0.15 + 0.08 * float(min(4, area_xy_s)) + m_rot_tie)
                                    if is_curved_step_tie
                                    else ((vd_norm_s / (ev_norm_s ** 0.55)) + 0.42 * vd_norm_s + m_rot_tie)
                                )
                                all_sl_cands.append(
                                    (
                                        sc_sm,
                                        0.0 if is_curved_step_tie else val_diff,
                                        pid_sm,
                                        LDUFrame(rid_sm, pos_sm),
                                        p_sl0,
                                        h_pl_s,
                                        rx0_s,
                                        rx1_s,
                                        ry0_s,
                                        ry1_s,
                                        top_stud_xy_s,
                                        banti_xy_s,
                                        clash_xy_s,
                                        tuple(sh_clear_list),
                                        occ_cells_s,
                                    )
                                )

        reserved_mold_counts: Dict[str, int] = {}
        rep_pen_sl = 0.08

        def _mold_s3_first_bonus(pid_b: str, vd_b: float) -> float:
            ent_b = CATALOG_RETRIEVAL_INDEX.get_entry(pid_b)
            if ent_b is not None and ent_b.taxonomy_class in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "THIN_PANEL"):
                return 0.12 if vd_b <= 1e-6 else 0.10
            return 0.05

        sl_heap = [
            (
                -(c[0] + _mold_s3_first_bonus(c[2], c[1])),
                -c[1],
                0,
                c[2],
                c[3].rot_id,
                c[3].pos,
                c[3],
                c[4],
                c[5],
                c[6],
                c[7],
                c[8],
                c[9],
                c[10],
                c[11],
                c[12],
                c[13],
                c[14],
            )
            for c in all_sl_cands
        ]
        heapq.heapify(sl_heap)
        sl_res_zb = [[0] * w_y for _ in range(w_x)]
        sm_tile_zb = [[sum((1 << p_k) for p_k, v_k in enumerate(col_k) if v_k) for col_k in row_k] for row_k in must_be_smooth_tile_occ.tolist()]
        while sl_heap:
            (
                neg_sc,
                neg_vd,
                used_seen,
                pid_cs,
                rid_cs_k,
                pos_cs_k,
                pose_cs,
                p_cs,
                h_cs,
                rx0_c,
                rx1_c,
                ry0_c,
                ry1_c,
                tstud_c,
                banti_c,
                clash_c,
                sh_clear_c,
                occ_cells_c,
            ) = heapq.heappop(sl_heap)
            cur_used = reserved_mold_counts.get(pid_cs, 0)
            if -neg_vd <= 1e-6 and cur_used >= 1:
                continue
            if cur_used != used_seen:
                fb_s = _mold_s3_first_bonus(pid_cs, -neg_vd)
                base_sc = -neg_sc - (fb_s if used_seen == 0 else 0.0) + rep_pen_sl * float(min(12, used_seen))
                heapq.heappush(
                    sl_heap,
                    (
                        -(base_sc + (fb_s if cur_used == 0 else 0.0) - rep_pen_sl * float(min(12, cur_used))),
                        neg_vd,
                        cur_used,
                        pid_cs,
                        rid_cs_k,
                        pos_cs_k,
                        pose_cs,
                        p_cs,
                        h_cs,
                        rx0_c,
                        rx1_c,
                        ry0_c,
                        ry1_c,
                        tstud_c,
                        banti_c,
                        clash_c,
                        sh_clear_c,
                        occ_cells_c,
                    ),
                )
                continue
            if rem_inv is not None and rem_inv.get(pid_cs, 0) <= 0:
                continue
            z_cs_m = ((1 << h_cs) - 1) << p_cs
            if any(
                (sl_res_zb[ux_o][uy_o] | inv_zb[ux_o][uy_o] | sm_tile_zb[ux_o][uy_o]) & z_cs_m
                for ux_o, uy_o in occ_cells_c
            ):
                continue
            if sh_clear_c and any(
                ((sl_res_zb[ux_s][uy_s] | inv_zb[ux_s][uy_s] | sm_tile_zb[ux_s][uy_s]) & (1 << p_s)) != 0
                or ((p_s + 1) < n_plates_z and (sl_res_zb[ux_s][uy_s] >> (p_s + 1)) != 0)
                for ux_s, uy_s, p_s in sh_clear_c
            ):
                continue
            # Dynamic full downward support check against already-reserved lower slopes
            smooth_cap_needed_xy: List[Tuple[int, int]] = []
            if p_cs > 0:
                if (p_cs - 1) in jumper_top_stud_xy:
                    sup_dyn = jumper_top_stud_xy[p_cs - 1]
                elif layer_phase[p_cs - 1] == layer_phase[p_cs]:
                    sup_dyn = has_top_stud_at_p[:, :, p_cs - 1]
                else:
                    sup_dyn = _support_mask_at_layer(p_cs)
                if sup_dyn is not None:
                    if (p_cs - 1) in jumper_top_stud_xy:
                        dyn_ok = (
                            sum(1 for bx_d, by_d in banti_c if sup_dyn[bx_d, by_d])
                            >= max(1, (len(banti_c) + 1) // 2)
                        ) and not any(sup_dyn[bx_d, by_d] for bx_d, by_d in clash_c if (bx_d, by_d) in occ_cells_c)
                    else:
                        dyn_ok = True
                        bit_csm1 = 1 << (p_cs - 1)
                        for bx_d, by_d in occ_cells_c:
                            if (bx_d, by_d) in banti_c:
                                if not sup_dyn[bx_d, by_d]:
                                    dyn_ok = False
                                    break
                            elif (bx_d, by_d) in clash_c and has_top_stud_at_p[bx_d, by_d, p_cs - 1]:
                                can_cap_now = (
                                    has_3070b_for_corner
                                    and (core_zb[bx_d][by_d] & bit_csm1) != 0
                                    and ((sl_res_zb[bx_d][by_d] | inv_zb[bx_d][by_d]) & bit_csm1) == 0
                                    and (p_cs == 1 or bool(has_top_stud_at_p[bx_d, by_d, p_cs - 2]))
                                )
                                if can_cap_now:
                                    smooth_cap_needed_xy.append((bx_d, by_d))
                                else:
                                    dyn_ok = False
                                    break
                    if not dyn_ok:
                        continue
            # Dynamic upward support check: never starve an already-reserved upper piece, jumper, or non-cleared `core_occ`
            sh_clear_set_c = set(sh_clear_c)
            if (p_cs + h_cs) < n_plates_z:
                up_starved = False
                p_top_c = p_cs + h_cs
                bit_top_c = 1 << p_top_c
                j_req_c = jumper_used_low.get(p_top_c)
                for ux_d, uy_d in occ_cells_c:
                    needs_stud = (
                        bool(j_req_c[ux_d, uy_d])
                        if j_req_c is not None
                        else bool(
                            (
                                (core_zb[ux_d][uy_d] & bit_top_c) != 0
                                and (ux_d, uy_d, p_top_c) not in sh_clear_set_c
                            )
                            or ((sl_res_zb[ux_d][uy_d] | inv_zb[ux_d][uy_d]) & bit_top_c) != 0
                        )
                    )
                    if needs_stud and (ux_d, uy_d) not in tstud_c:
                        up_starved = True
                        break
                if up_starved:
                    continue

            for ux_s, uy_s, p_s in sh_clear_c:
                core_occ[ux_s, uy_s, p_s] = False
                core_zb[ux_s][uy_s] &= ~(1 << p_s)
                has_top_stud_at_p[ux_s, uy_s, p_s] = False

            for sc_x, sc_y in smooth_cap_needed_xy:
                must_be_smooth_tile_occ[sc_x, sc_y, p_cs - 1] = True
                sm_tile_zb[sc_x][sc_y] |= 1 << (p_cs - 1)
                has_top_stud_at_p[sc_x, sc_y, p_cs - 1] = False

            for ux_o, uy_o in occ_cells_c:
                slope_reserved_occ[ux_o, uy_o, p_cs : p_cs + h_cs] = True
                sl_res_zb[ux_o][uy_o] |= z_cs_m
                has_top_stud_at_p[ux_o, uy_o, p_cs : p_cs + h_cs] = False
            for sx_t, sy_t in tstud_c:
                if rx0_c <= sx_t < rx1_c and ry0_c <= sy_t < ry1_c:
                    has_top_stud_at_p[sx_t, sy_t, p_cs + h_cs - 1] = True
            planned_by_z.setdefault(p_cs * 8, []).append((pid_cs, pose_cs))
            reserved_mold_counts[pid_cs] = cur_used + 1
            if rem_inv is not None:
                rem_inv[pid_cs] = rem_inv.get(pid_cs, 0) - 1

    # Cap any remaining exposed terrace cells alongside `15573` jumper bridge layers with smooth tiles
    for p_jb in sorted(jumper_bridge_layers.keys()):
        ph_from_jb, _ = jumper_bridge_layers[p_jb]
        z0_jb = p_jb * 8
        z6_t0 = min(occ6.shape[2], int(math.ceil(z0_jb / 6.0)))
        z6_t1 = min(occ6.shape[2], int(math.ceil((z0_jb + 8) / 6.0)))
        if z6_t1 <= z6_t0:
            continue
        low_sup_jb = (
            has_top_stud_at_p[:, :, p_jb - 1]
            if (p_jb > 0 and (p_jb - 1) not in jumper_bridge_layers)
            else jumper_low_occ[p_jb]
        )
        u_low_jb = jumper_used_low[p_jb]
        cg_ter = phase_grids[ph_from_jb][:, :, p_jb]
        terrace_occ = (cg_ter >= 42) & low_sup_jb & (~u_low_jb) & (~slope_reserved_occ[:, :, p_jb])
        for tx_jb in range(w_x):
            for ty_jb in range(w_y):
                if terrace_occ[tx_jb, ty_jb]:
                    x6_tb = (tx_jb * 20 + ph_from_jb[0]) // 5
                    y6_tb = (ty_jb * 20 + ph_from_jb[1]) // 5
                    tb_sl = occ6[x6_tb : x6_tb + 4, y6_tb : y6_tb + 4, z6_t0:z6_t1]
                    if tb_sl.size == 0 or 2 * int(np.count_nonzero(tb_sl)) < int(tb_sl.size):
                        terrace_occ[tx_jb, ty_jb] = False
        if np.any(terrace_occ) and tile_templates:
            t_pls = _tile_boolean_layer_with_rects(
                terrace_occ,
                z0=z0_jb,
                ox=ph_from_jb[0],
                oy=ph_from_jb[1],
                templates=tile_templates,
                prefer_x_major=(p_jb % 2 == 1),
                parity_shift=((p_jb // 2) % 2),
                rem_inv=rem_inv,
                supported_below_xy=low_sup_jb & (~u_low_jb),
            )
            if t_pls:
                planned_by_z.setdefault(z0_jb, []).extend(t_pls)

    rem_core = core_occ & (~inv_reserved_occ) & (~slope_reserved_occ)

    # Place flush `4 LDU`-thin vertical perimeter brackets (`44728`, `99780`, `99207` — all `CC-BY-4.0`)
    # on `1x2` perimeter plate runs whenever a `4 LDU` (`1` sub-voxel) outer wall/taper slice strictly improves `occ6` IoU!
    bracket_cands = [bpid for bpid in ("44728", "99780", "99207") if bpid in kit_set]
    if bracket_cands:
        planned_blade_boxes: List[LDUBox] = []
        for bpid in bracket_cands:
            prim_br = get_brick_primitive(bpid)
            if len(prim_br.hull_boxes) != 2:
                continue
            for p_br in range(n_plates_z):
                if p_br in jumper_bridge_layers or not np.any(rem_core[:, :, p_br]):
                    continue
                ph_br = layer_phase[p_br]
                sup_br = _support_mask_at_layer(p_br)
                z0_br = p_br * 8
                for rid_br in C4_Z_ROT_IDS:
                    if rem_inv is not None and rem_inv.get(bpid, 0) <= 0:
                        break
                    b_base = prim_br.hull_boxes[0].transform(LDUFrame(rid_br, (0, 0, 0)))
                    b_blade = prim_br.hull_boxes[1].transform(LDUFrame(rid_br, (0, 0, 0)))
                    wx_br = (b_base.xmax - b_base.xmin) // 20
                    wy_br = (b_base.ymax - b_base.ymin) // 20
                    if wx_br <= 0 or wy_br <= 0:
                        continue
                    odx_b = 1 if b_blade.xmin >= b_base.xmax else (-1 if b_blade.xmax <= b_base.xmin else 0)
                    ody_b = 1 if b_blade.ymin >= b_base.ymax else (-1 if b_blade.ymax <= b_base.ymin else 0)
                    if abs(odx_b) + abs(ody_b) != 1:
                        continue
                    zb_min = z0_br - b_base.zmin + b_blade.zmin
                    zb_max = z0_br - b_base.zmin + b_blade.zmax
                    if zb_min < 0 or int(math.ceil(zb_max / 6.0)) > gz6_nonempty_limit:
                        continue
                    p_chk0 = max(0, zb_min // 8)
                    p_chk1 = n_plates_z if zb_min < z0_br else min(n_plates_z, int(math.ceil(zb_max / 8.0)))
                    for ix_b in range(w_x - wx_br + 1):
                        for iy_b in range(w_y - wy_br + 1):
                            if rem_inv is not None and rem_inv.get(bpid, 0) <= 0:
                                break
                            if not np.all(rem_core[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br, p_br]):
                                continue
                            if np.any(must_be_smooth_tile_occ[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br, p_br]):
                                continue
                            if sup_br is not None and not np.all(sup_br[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br]):
                                continue
                            # For upward blade (`zb_max > z0_br + 8`), avoid crossing phase-shift layers above `p_br`
                            if zb_max > z0_br + 8 and any(
                                layer_phase[pq_b] != ph_br or pq_b in jumper_bridge_layers
                                for pq_b in range(p_br, min(n_plates_z, zb_max // 8))
                            ):
                                continue
                            # Ensure outer cell column is clear across `p_chk0 .. p_chk1` (and below `p_br` for downward blades)
                            outer_clear = True
                            for dx_k in range(wx_br):
                                for dy_k in range(wy_br):
                                    ox_k = ix_b + dx_k + odx_b
                                    oy_k = iy_b + dy_k + ody_b
                                    if 0 <= ox_k < w_x and 0 <= oy_k < w_y:
                                        if (
                                            np.any(core_occ[ox_k, oy_k, p_chk0:p_chk1])
                                            or np.any(inv_reserved_occ[ox_k, oy_k, p_chk0:p_chk1])
                                            or np.any(slope_reserved_occ[ox_k, oy_k, p_chk0:p_chk1])
                                        ):
                                            outer_clear = False
                                            break
                                if not outer_clear:
                                    break
                            if not outer_clear:
                                continue
                            fx0_b = ix_b * 20 + ph_br[0]
                            fy0_b = iy_b * 20 + ph_br[1]
                            pose_br = LDUFrame(rid_br, (fx0_b - b_base.xmin, fy0_b - b_base.ymin, z0_br - b_base.zmin))
                            wb_base_w = prim_br.hull_boxes[0].transform(pose_br)
                            wb_blade_w = prim_br.hull_boxes[1].transform(pose_br)
                            if (
                                _box_hits_forbidden(wb_base_w, forbidden_support_boxes)
                                or _box_hits_forbidden(wb_blade_w, forbidden_support_boxes)
                                or any(wb_blade_w.intersects(eb) for eb in planned_blade_boxes)
                            ):
                                continue
                            bx0_6 = int(math.floor(wb_blade_w.xmin / 5.0))
                            bx1_6 = int(math.ceil(wb_blade_w.xmax / 5.0))
                            by0_6 = int(math.floor(wb_blade_w.ymin / 5.0))
                            by1_6 = int(math.ceil(wb_blade_w.ymax / 5.0))
                            bz0_6 = int(math.floor(wb_blade_w.zmin / 6.0))
                            bz1_6 = int(math.ceil(wb_blade_w.zmax / 6.0))
                            if (
                                bx0_6 < 0
                                or by0_6 < 0
                                or bz0_6 < 0
                                or bx1_6 > occ6.shape[0]
                                or by1_6 > occ6.shape[1]
                                or bz1_6 > occ6.shape[2]
                            ):
                                continue
                            bl_sl = occ6[bx0_6:bx1_6, by0_6:by1_6, bz0_6:bz1_6]
                            bl_tot = int(bl_sl.size)
                            bl_tp = int(np.count_nonzero(bl_sl))
                            bl_fp = bl_tot - bl_tp
                            min_bl_tp = 8 if is_thin_regime else 18
                            if bl_tp < min_bl_tp or bl_tp < (bl_fp + 2 if is_thin_regime else 2 * bl_fp + 4):
                                continue
                            planned_by_z.setdefault(z0_br, []).append((bpid, pose_br))
                            planned_blade_boxes.append(wb_blade_w)
                            rem_core[ix_b : ix_b + wx_br, iy_b : iy_b + wy_br, p_br] = False
                            if rem_inv is not None:
                                rem_inv[bpid] = rem_inv.get(bpid, 0) - 1

    # Fuse aligned 3-plate (`24 LDU`) runs where phase is identical, all 3 layers are True, and top isn't reserved smooth-tile
    p_idx = 0
    while p_idx + 2 < n_plates_z:
        if (
            p_idx not in jumper_bridge_layers
            and (p_idx + 1) not in jumper_bridge_layers
            and (p_idx + 2) not in jumper_bridge_layers
            and layer_phase[p_idx] == layer_phase[p_idx + 1] == layer_phase[p_idx + 2]
        ):
            fuse_mask = (
                rem_core[:, :, p_idx]
                & rem_core[:, :, p_idx + 1]
                & rem_core[:, :, p_idx + 2]
                & (~must_be_smooth_tile_occ[:, :, p_idx])
                & (~must_be_smooth_tile_occ[:, :, p_idx + 1])
                & (~must_be_smooth_tile_occ[:, :, p_idx + 2])
            )
            sup_f_chk = _support_mask_at_layer(p_idx)
            if sup_f_chk is not None and bool(
                np.any(rem_core[:, :, p_idx] & (~sup_f_chk) & (~fuse_mask))
            ):
                p_idx += 1
                continue
            if np.any(fuse_mask):
                ph_f = layer_phase[p_idx]
                c_idx = p_idx // 3
                sup_f = sup_f_chk
                z0_f = p_idx * 8
                z6_f0 = max(0, z0_f // 6)
                z6_f1 = min(occ6.shape[2], (z0_f + 24) // 6)
                # Seed protrusion-safe calibrated straight-edge (`30414`, `11211`, `87087`)
                # outward SNOT core bricks across ALL 4 carved frustum/terrace walls!
                snot_core_cands = [spid for spid in ("30414", "11211", "87087") if spid in kit_set]
                if snot_core_cands:
                    def _cell_dir_snot_hits(cx_i: int, cy_i: int, odx: int, ody: int) -> int:
                        ox_c, oy_c = cx_i + odx, cy_i + ody
                        if 0 <= ox_c < w_x and 0 <= oy_c < w_y:
                            if (
                                np.any(core_occ[ox_c, oy_c, p_idx : p_idx + 3])
                                or np.any(inv_reserved_occ[ox_c, oy_c, p_idx : p_idx + 3])
                                or np.any(slope_reserved_occ[ox_c, oy_c, p_idx : p_idx + 3])
                            ):
                                return 0
                        fx0_c = cx_i * 20 + ph_f[0]
                        fx1_c = (cx_i + 1) * 20 + ph_f[0]
                        fy0_c = cy_i * 20 + ph_f[1]
                        fy1_c = (cy_i + 1) * 20 + ph_f[1]
                        if odx == 1:
                            mx0, mx1, my0, my1 = fx1_c // 5, fx1_c // 5 + 2, fy0_c // 5, fy1_c // 5
                        elif odx == -1:
                            mx0, mx1, my0, my1 = fx0_c // 5 - 2, fx0_c // 5, fy0_c // 5, fy1_c // 5
                        elif ody == 1:
                            mx0, mx1, my0, my1 = fx0_c // 5, fx1_c // 5, fy1_c // 5, fy1_c // 5 + 2
                        else:
                            mx0, mx1, my0, my1 = fx0_c // 5, fx1_c // 5, fy0_c // 5 - 2, fy0_c // 5
                        if mx0 < 0 or my0 < 0 or mx1 > occ6.shape[0] or my1 > occ6.shape[1]:
                            return 0
                        m_strip = occ6[mx0:mx1, my0:my1, z6_f0:z6_f1]
                        n_hits_s = int(np.count_nonzero(m_strip))
                        if m_strip.size == 0 or n_hits_s < int(0.18 * m_strip.size):
                            return 0
                        prot_box = LDUBox(
                            min(fx0_c, mx0 * 5),
                            min(fy0_c, my0 * 5),
                            z0_f,
                            max(fx1_c, mx1 * 5),
                            max(fy1_c, my1 * 5),
                            z0_f + 24,
                        )
                        return n_hits_s if not _box_hits_forbidden(prot_box, forbidden_support_boxes) else 0

                    dir_list_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
                    cell_valid_dirs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
                    for ix_d in range(w_x):
                        for iy_d in range(w_y):
                            if not fuse_mask[ix_d, iy_d]:
                                continue
                            if sup_f is not None and not sup_f[ix_d, iy_d]:
                                continue
                            vds_hits = [
                                (h_v, d_v)
                                for d_v in dir_list_4
                                if (h_v := _cell_dir_snot_hits(ix_d, iy_d, d_v[0], d_v[1])) > 0
                            ]
                            if vds_hits:
                                if is_thin_regime:
                                    vds_hits.sort(key=lambda item: -item[0])
                                    cell_valid_dirs[(ix_d, iy_d)] = [d_v for _, d_v in vds_hits]
                                else:
                                    vds = [d_v for _, d_v in vds_hits]
                                    rot_d = (ix_d + iy_d + c_idx) % len(vds)
                                    cell_valid_dirs[(ix_d, iy_d)] = vds[rot_d:] + vds[:rot_d]

                    # Pass 1: Place multi-stud `30414` (`1x4`) and `11211` (`1x2`) ONLY along interior straight wall
                    # segments (`len(cell_valid_dirs) == 1`) so 2-exposed corner cells remain free to alternate walls!
                    for spid in [s_m for s_m in ("30414", "11211") if s_m in snot_core_cands]:
                        prim_sn = get_brick_primitive(spid)
                        lat_ps = [pt for pt in prim_sn.ports if pt.polarity == PortPolarity.STUD and pt.local_frame.z_axis[2] == 0]
                        if not lat_ps:
                            continue
                        for rid_sn in C4_Z_ROT_IDS:
                            if rem_inv is not None and rem_inv.get(spid, 0) <= 0:
                                break
                            b_sn = prim_sn.hull_boxes[0].transform(LDUFrame(rid_sn, (0, 0, 0)))
                            wx_sn = (b_sn.xmax - b_sn.xmin) // 20
                            wy_sn = (b_sn.ymax - b_sn.ymin) // 20
                            f_norm = LDUFrame(rid_sn, (-b_sn.xmin, -b_sn.ymin, -b_sn.zmin))
                            wf_0 = f_norm.compose(lat_ps[0].local_frame)
                            odx_m, ody_m = int(wf_0.z_axis[0]), int(wf_0.z_axis[1])
                            for ix_s in range(w_x - wx_sn + 1):
                                for iy_s in range(w_y - wy_sn + 1):
                                    if rem_inv is not None and rem_inv.get(spid, 0) <= 0:
                                        break
                                    if not np.all(fuse_mask[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn]):
                                        continue
                                    if sup_f is not None and not np.all(sup_f[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn]):
                                        continue
                                    # Require every covered cell to need `(odx_m, ody_m)`; allow at most 1 corner on 4-stud `30414` on wide footprints
                                    seg_ok = True
                                    corner_used_sn = 0
                                    max_corners_sn = 1 if (wx_sn * wy_sn >= 4 and w_x * w_y >= 24) else 0
                                    for dx_k in range(wx_sn):
                                        for dy_k in range(wy_sn):
                                            v_k = cell_valid_dirs.get((ix_s + dx_k, iy_s + dy_k), [])
                                            if (odx_m, ody_m) not in v_k:
                                                seg_ok = False
                                                break
                                            if len(v_k) > 1:
                                                corner_used_sn += 1
                                                if corner_used_sn > max_corners_sn:
                                                    seg_ok = False
                                                    break
                                        if not seg_ok:
                                            break
                                    if not seg_ok:
                                        continue
                                    fx0_l = ix_s * 20 + ph_f[0]
                                    fy0_l = iy_s * 20 + ph_f[1]
                                    pose_sn = LDUFrame(rid_sn, (fx0_l - b_sn.xmin, fy0_l - b_sn.ymin, z0_f - b_sn.zmin))
                                    planned_by_z.setdefault(z0_f, []).append((spid, pose_sn))
                                    fuse_mask[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn] = False
                                    rem_core[ix_s : ix_s + wx_sn, iy_s : iy_s + wy_sn, p_idx : p_idx + 3] = False
                                    if rem_inv is not None:
                                        rem_inv[spid] = rem_inv.get(spid, 0) - 1

                    # Pass 2: Place `87087` (`1x1` SNOT brick) at every remaining perimeter/corner cell in `cell_valid_dirs`,
                    # alternating preferred direction `(ix + iy + c_idx) % len(vds)` so ALL 4 carved walls get lateral studs!
                    if "87087" in snot_core_cands:
                        prim_87 = get_brick_primitive("87087")
                        lat_87 = [pt for pt in prim_87.ports if pt.polarity == PortPolarity.STUD and pt.local_frame.z_axis[2] == 0]
                        dir_to_87: Dict[Tuple[int, int], Tuple[int, LDUBox]] = {}
                        if lat_87:
                            for rid_87 in C4_Z_ROT_IDS:
                                b_87 = prim_87.hull_boxes[0].transform(LDUFrame(rid_87, (0, 0, 0)))
                                wf_87 = LDUFrame(rid_87, (-b_87.xmin, -b_87.ymin, -b_87.zmin)).compose(lat_87[0].local_frame)
                                dir_to_87[(int(wf_87.z_axis[0]), int(wf_87.z_axis[1]))] = (rid_87, b_87)
                        for (ix_s, iy_s), vds_c in sorted(cell_valid_dirs.items()):
                            if rem_inv is not None and rem_inv.get("87087", 0) <= 0:
                                break
                            if not fuse_mask[ix_s, iy_s]:
                                continue
                            pref_dir = vds_c[0]
                            if pref_dir not in dir_to_87:
                                continue
                            rid_87, b_87 = dir_to_87[pref_dir]
                            fx0_l = ix_s * 20 + ph_f[0]
                            fy0_l = iy_s * 20 + ph_f[1]
                            pose_87 = LDUFrame(rid_87, (fx0_l - b_87.xmin, fy0_l - b_87.ymin, z0_f - b_87.zmin))
                            planned_by_z.setdefault(z0_f, []).append(("87087", pose_87))
                            fuse_mask[ix_s, iy_s] = False
                            rem_core[ix_s, iy_s, p_idx : p_idx + 3] = False
                            if rem_inv is not None:
                                rem_inv["87087"] = rem_inv.get("87087", 0) - 1
                if (p_idx - 1) in jumper_bridge_layers:
                    j_from, j_to = jumper_bridge_layers[p_idx - 1]
                    pref_x = j_to[0] != j_from[0]
                    par_sh = 0
                else:
                    pref_x = c_idx % 2 == 0
                    par_sh = c_idx % 2
                b_pls = _tile_boolean_layer_with_rects(
                    fuse_mask,
                    z0=z0_f,
                    ox=ph_f[0],
                    oy=ph_f[1],
                    templates=brick_templates,
                    prefer_x_major=pref_x,
                    parity_shift=par_sh,
                    rem_inv=rem_inv,
                    supported_below_xy=sup_f,
                )
                if b_pls:
                    planned_by_z.setdefault(z0_f, []).extend(b_pls)
                    for pid_b, pose_b in b_pls:
                        wb_b = get_brick_primitive(pid_b).hull_boxes[0].transform(pose_b)
                        ix0 = (wb_b.xmin - ph_f[0]) // 20
                        ix1 = (wb_b.xmax - ph_f[0]) // 20
                        iy0 = (wb_b.ymin - ph_f[1]) // 20
                        iy1 = (wb_b.ymax - ph_f[1]) // 20
                        rem_core[ix0:ix1, iy0:iy1, p_idx : p_idx + 3] = False
                p_idx += 3
                continue
        p_idx += 1

    # Fill interior solid mesh cavities around `15573` jumper bridges at `p in jumper_bridge_layers`
    # (`using studless smooth tiles at `p*8`, or `8 LDU plate + 8 LDU smooth tile` across 2-layer bridges)
    if jumper_bridge_layers and tile_templates and not is_thin_regime:
        all_planned_boxes: List[LDUBox] = [
            b_pl.transform(pose_pl)
            for z_list in planned_by_z.values()
            for pid_pl, pose_pl in z_list
            for b_pl in get_brick_primitive(pid_pl).hull_boxes
        ]
        for p_jb_f in sorted(jumper_bridge_layers.keys()):
            if p_jb_f <= 0 or (p_jb_f - 1) in jumper_bridge_layers:
                continue
            ph_low_f, _ = jumper_bridge_layers[p_jb_f]
            z0_jb = p_jb_f * 8
            z6_jb0 = min(occ6.shape[2], int(math.ceil(z0_jb / 6.0)))
            z6_jb1 = min(occ6.shape[2], int(math.ceil((z0_jb + 8) / 6.0)))
            if z6_jb1 <= z6_jb0:
                continue
            is_two_jb = (p_jb_f + 1) in jumper_bridge_layers
            z6_jb2 = min(occ6.shape[2], int(math.ceil((z0_jb + 16) / 6.0)))
            fill_1_mask = np.zeros((w_x, w_y), dtype=bool)
            fill_2_mask = np.zeros((w_x, w_y), dtype=bool)
            for ix_f in range(w_x):
                for iy_f in range(w_y):
                    if not has_top_stud_at_p[ix_f, iy_f, p_jb_f - 1] or jumper_used_low[p_jb_f][ix_f, iy_f]:
                        continue
                    bx0_f = ix_f * 20 + ph_low_f[0]
                    by0_f = iy_f * 20 + ph_low_f[1]
                    wb1_f = LDUBox(bx0_f, by0_f, z0_jb, bx0_f + 20, by0_f + 20, z0_jb + 8)
                    if _box_hits_forbidden(wb1_f, forbidden_support_boxes):
                        continue
                    if any(
                        b_ex.xmin < wb1_f.xmax
                        and b_ex.xmax > wb1_f.xmin
                        and b_ex.ymin < wb1_f.ymax
                        and b_ex.ymax > wb1_f.ymin
                        and b_ex.zmin < wb1_f.zmax
                        and b_ex.zmax > wb1_f.zmin
                        for b_ex in all_planned_boxes
                    ):
                        continue
                    sx6_f, sy6_f = bx0_f // 5, by0_f // 5
                    m1_sl = occ6[sx6_f : sx6_f + 4, sy6_f : sy6_f + 4, z6_jb0:z6_jb1]
                    if m1_sl.size == 0 or int(np.count_nonzero(m1_sl)) < int(0.58 * m1_sl.size):
                        continue
                    if is_two_jb and plate_templates and z6_jb2 > z6_jb1:
                        wb2_f = LDUBox(bx0_f, by0_f, z0_jb + 8, bx0_f + 20, by0_f + 20, z0_jb + 16)
                        m2_sl = occ6[sx6_f : sx6_f + 4, sy6_f : sy6_f + 4, z6_jb1:z6_jb2]
                        clear_2 = (
                            not _box_hits_forbidden(wb2_f, forbidden_support_boxes)
                            and m2_sl.size > 0
                            and int(np.count_nonzero(m2_sl)) >= int(0.62 * m2_sl.size)
                            and not any(
                                b_ex.xmin < wb2_f.xmax
                                and b_ex.xmax > wb2_f.xmin
                                and b_ex.ymin < wb2_f.ymax
                                and b_ex.ymax > wb2_f.ymin
                                and b_ex.zmin < wb2_f.zmax
                                and b_ex.zmax > wb2_f.zmin
                                for b_ex in all_planned_boxes
                            )
                        )
                        if clear_2:
                            fill_2_mask[ix_f, iy_f] = True
                            continue
                    fill_1_mask[ix_f, iy_f] = True
            if np.any(fill_2_mask):
                pls_jb_p0 = _tile_boolean_layer_with_rects(
                    fill_2_mask,
                    z0=z0_jb,
                    ox=ph_low_f[0],
                    oy=ph_low_f[1],
                    templates=plate_templates,
                    prefer_x_major=True,
                    parity_shift=0,
                    rem_inv=rem_inv,
                    supported_below_xy=fill_2_mask,
                )
                pls_jb_t1 = _tile_boolean_layer_with_rects(
                    fill_2_mask,
                    z0=z0_jb + 8,
                    ox=ph_low_f[0],
                    oy=ph_low_f[1],
                    templates=tile_templates,
                    prefer_x_major=False,
                    parity_shift=0,
                    rem_inv=rem_inv,
                    supported_below_xy=fill_2_mask,
                )
                if pls_jb_p0 and pls_jb_t1:
                    planned_by_z.setdefault(z0_jb, []).extend(pls_jb_p0)
                    planned_by_z.setdefault(z0_jb + 8, []).extend(pls_jb_t1)
                    for pid_n, pose_n in pls_jb_p0 + pls_jb_t1:
                        all_planned_boxes.extend(b_n.transform(pose_n) for b_n in get_brick_primitive(pid_n).hull_boxes)
            if np.any(fill_1_mask):
                pls_jb_t0 = _tile_boolean_layer_with_rects(
                    fill_1_mask,
                    z0=z0_jb,
                    ox=ph_low_f[0],
                    oy=ph_low_f[1],
                    templates=tile_templates,
                    prefer_x_major=True,
                    parity_shift=0,
                    rem_inv=rem_inv,
                    supported_below_xy=fill_1_mask,
                )
                if pls_jb_t0:
                    planned_by_z.setdefault(z0_jb, []).extend(pls_jb_t0)
                    for pid_n, pose_n in pls_jb_t0:
                        all_planned_boxes.extend(b_n.transform(pose_n) for b_n in get_brick_primitive(pid_n).hull_boxes)

    # Tile remaining `8 LDU` plate layers (tiling `must_be_smooth_tile_occ` corner cells with `3070b` smooth tiles!)
    for p in range(n_plates_z):
        if p in jumper_bridge_layers:
            continue
        z0 = p * 8
        if np.any(rem_core[:, :, p]):
            ph_p = layer_phase[p]
            sup_p = _support_mask_at_layer(p)
            sm_cells_p = rem_core[:, :, p] & must_be_smooth_tile_occ[:, :, p]
            if np.any(sm_cells_p) and tile_templates:
                t_corner_pls = _tile_boolean_layer_with_rects(
                    sm_cells_p,
                    z0=z0,
                    ox=ph_p[0],
                    oy=ph_p[1],
                    templates=[t for t in tile_templates if t[1] == 1 and t[2] == 1] or tile_templates,
                    prefer_x_major=(p % 2 == 1),
                    parity_shift=((p // 2) % 2),
                    rem_inv=rem_inv,
                    supported_below_xy=sup_p,
                )
                if t_corner_pls:
                    planned_by_z.setdefault(z0, []).extend(t_corner_pls)
            reg_cells_p = rem_core[:, :, p] & (~must_be_smooth_tile_occ[:, :, p])
            if np.any(reg_cells_p):
                if (p - 1) in jumper_bridge_layers:
                    jp_from, jp_to = jumper_bridge_layers[p - 1]
                    pref_xp = jp_to[0] != jp_from[0]
                    par_p = 0
                else:
                    pref_xp = p % 2 == 1
                    par_p = (p // 2) % 2
                p_pls = _tile_boolean_layer_with_rects(
                    reg_cells_p,
                    z0=z0,
                    ox=ph_p[0],
                    oy=ph_p[1],
                    templates=plate_templates,
                    prefer_x_major=pref_xp,
                    parity_shift=par_p,
                    rem_inv=rem_inv,
                    supported_below_xy=sup_p,
                )
                if p_pls:
                    planned_by_z.setdefault(z0, []).extend(p_pls)

    # Commit planned core bottom-up while maintaining live `occ6_live`, incremental 6-DoF witness,
    # monotone insertion corridors, and `2 LDU` CSG sets!
    state = AssemblyState()
    if inventory_counts is not None:
        rem_inv = dict(inventory_counts)

    occ6_live = np.zeros_like(occ6)
    inter_live = 0
    mesh_sub_total = int(np.count_nonzero(occ6))
    over_live = 0  # `union_live = mesh_sub_total + over_live`!

    base_witness = None
    base_lp_eta = 0.0
    plumb_deltas: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    plumb_hop_cache: Dict[int, Any] = {}

    def _add_boxes_to_live_occ6(wbs: Sequence[LDUBox]) -> None:
        nonlocal inter_live, over_live
        gx, gy, gz = occ6.shape
        for b in wbs:
            x0 = max(0, int(math.floor(b.xmin / 5.0)))
            x1 = min(gx, int(math.ceil(b.xmax / 5.0)))
            y0 = max(0, int(math.floor(b.ymin / 5.0)))
            y1 = min(gy, int(math.ceil(b.ymax / 5.0)))
            z0 = max(0, int(math.floor(b.zmin / 6.0)))
            z1 = min(gz, int(math.ceil(b.zmax / 6.0)))
            if x1 > x0 and y1 > y0 and z1 > z0:
                cur_sl = occ6_live[x0:x1, y0:y1, z0:z1]
                new_m = ~cur_sl
                if np.any(new_m):
                    tgt_sl = occ6[x0:x1, y0:y1, z0:z1]
                    inter_live += int(np.count_nonzero(new_m & tgt_sl))
                    over_live += int(np.count_nonzero(new_m & (~tgt_sl)))
                    cur_sl[:] = True

    thin_panel_pids: Set[str] = {
        pid_tp
        for pid_tp in kit_ids
        if (ent_tp := CATALOG_RETRIEVAL_INDEX.get_entry(pid_tp)) is not None
        and ent_tp.taxonomy_class == "THIN_PANEL"
    }

    def _eval_fast_pose_delta_iou(
        pid_f: str,
        rid_f: int,
        wx_f: int,
        wy_f: int,
        wz_f: int,
        require_ground_z0: bool = False,
        enforce_min_precision: bool = True,
    ) -> float:
        """
        Ultra-fast (`< 0.5 us` rejected / `~1 us` scored) zero-allocation exact `ΔIoU` pre-filter using
        `@lru_cache` relative `5x5x6 LDU` masks BEFORE constructing `LDUBox` tuples or running collision queries.
        """
        if mesh_sub_total == 0:
            return -1.0
        qx, rx_mod = divmod(wx_f, 5)
        qy, ry_mod = divmod(wy_f, 5)
        qz, rz_mod = divmod(wz_f, 6)
        rx0, rx1, ry0, ry1, rz0, rz1, min_z_loc, is_full, rmask = _fast_rel_mask6(
            pid_f, rid_f, rx_mod, ry_mod, rz_mod
        )
        w_zmin = min_z_loc + wz_f
        if require_ground_z0:
            if w_zmin != 0:
                return -1.0
        elif w_zmin < 0:
            return -1.0

        gx, gy, gz = occ6.shape
        x0, x1 = qx + rx0, qx + rx1
        y0, y1 = qy + ry0, qy + ry1
        z0, z1 = qz + rz0, qz + rz1
        # Never clip out-of-bounds overhangs: `occ6` already includes `+20 LDU` (`+4` sub-voxel) upper margin
        if x0 < 0 or y0 < 0 or z0 < 0 or x1 > gx or y1 > gy or z1 > gz:
            return -1.0
        if x1 <= x0 or y1 <= y0 or z1 <= z0 or z0 >= gz6_nonempty_limit:
            return -1.0
        box_tgt_cnt = int(
            occ6_csum[x1, y1, z1]
            - occ6_csum[x0, y1, z1]
            - occ6_csum[x1, y0, z1]
            - occ6_csum[x1, y1, z0]
            + occ6_csum[x0, y0, z1]
            + occ6_csum[x0, y1, z0]
            + occ6_csum[x1, y0, z0]
            - occ6_csum[x0, y0, z0]
        )
        if box_tgt_cnt == 0:
            return -1.0
        if require_ground_z0 and int(
            occ6_csum[x1, y1, 1]
            - occ6_csum[x0, y1, 1]
            - occ6_csum[x1, y0, 1]
            + occ6_csum[x0, y0, 1]
        ) < max(2, ((x1 - x0) * (y1 - y0)) // 4):
            return -1.0

        live_sl = occ6_live[x0:x1, y0:y1, z0:z1]
        if live_sl.all():
            return 0.0
        if is_full:
            live_cnt = int(np.count_nonzero(live_sl))
            tot_new = live_sl.size - live_cnt
            tgt_sl_f = occ6[x0:x1, y0:y1, z0:z1]
            if live_cnt == 0:
                d_in = box_tgt_cnt
                sky_ov_f = (
                    (x1 - x0) * (y1 - y0) * max(0, z1 - max(z0, gz6_nonempty_limit))
                    if z1 > gz6_nonempty_limit
                    else 0
                )
                if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
                    z_sk0 = max(0, eff_gz6_sky_limit - z0)
                    sk_tot = (x1 - x0) * (y1 - y0) * (z1 - max(z0, eff_gz6_sky_limit))
                    sk_in = int(np.count_nonzero(tgt_sl_f[:, :, z_sk0:]))
                    if sk_tot >= 8 and sk_in * 3 < sk_tot:
                        return -1.0
            else:
                new_m = ~live_sl
                d_in = int(np.count_nonzero(new_m & tgt_sl_f))
                if d_in == 0:
                    return -1.0
                sky_ov_f = (
                    int(np.count_nonzero(new_m[:, :, max(0, gz6_nonempty_limit - z0) :]))
                    if z1 > gz6_nonempty_limit
                    else 0
                )
                if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
                    z_sk0 = max(0, eff_gz6_sky_limit - z0)
                    sk_sl = new_m[:, :, z_sk0:]
                    sk_tot = int(np.count_nonzero(sk_sl))
                    if sk_tot >= 8 and int(np.count_nonzero(sk_sl & tgt_sl_f[:, :, z_sk0:])) * 3 < sk_tot:
                        return -1.0
            d_ov = tot_new - d_in
        else:
            new_m = rmask & (~live_sl)
            tot_new = int(np.count_nonzero(new_m))
            if tot_new == 0:
                return 0.0
            tgt_sl_nf = occ6[x0:x1, y0:y1, z0:z1]
            d_in = int(np.count_nonzero(new_m & tgt_sl_nf))
            if d_in == 0:
                return -1.0
            d_ov = tot_new - d_in
            sky_ov_f = (
                int(np.count_nonzero(new_m[:, :, max(0, gz6_nonempty_limit - z0) :]))
                if z1 > gz6_nonempty_limit
                else 0
            )
            if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
                z_sk0 = max(0, eff_gz6_sky_limit - z0)
                sk_sl = new_m[:, :, z_sk0:]
                sk_tot = int(np.count_nonzero(sk_sl))
                if sk_tot >= 8 and int(np.count_nonzero(sk_sl & tgt_sl_nf[:, :, z_sk0:])) * 3 < sk_tot:
                    return -1.0

        is_tp = pid_f in thin_panel_pids
        if enforce_min_precision and ((d_in * 3 < d_ov) if (is_tp or is_thin_regime) else (d_in * 2 < d_ov)):
            return -1.0
        if z1 > gz6_nonempty_limit and sky_ov_f * 2 > d_in:
            return -1.0

        # Guard against low-precision boulders during initial seeding and on thin wireframes
        if enforce_min_precision and inter_live < int(0.18 * mesh_sub_total) and float(d_in) <= 0.22 * float(d_ov):
            return -1.0
        is_sideways_thin_skin = (
            min(x1 - x0, y1 - y0) <= 3 and (z1 - z0) >= 3
        )
        if enforce_min_precision and is_thin_regime and not is_tp and not is_sideways_thin_skin:
            if tot_new >= 48 and float(d_in) < 0.85 * float(d_ov):
                return -1.0
            if tot_new >= 32 and float(d_in) < 0.55 * float(d_ov):
                return -1.0

        u_cur = float(max(1, mesh_sub_total + over_live))
        iou_cur = float(inter_live) / u_cur
        iou_new = float(inter_live + d_in) / float(max(1, mesh_sub_total + over_live + d_ov))
        return iou_new - iou_cur

    def _eval_boxes_delta_iou(
        wbs: Sequence[LDUBox],
        enforce_min_precision: bool = True,
    ) -> Tuple[float, int, int]:
        """Compute exact global `ΔIoU` on `occ6` in ~4 microseconds without mutating `occ6_live`."""
        if mesh_sub_total == 0:
            return -1.0, 0, 0
        gx, gy, gz = occ6.shape
        xmin = min(b.xmin for b in wbs)
        xmax = max(b.xmax for b in wbs)
        ymin = min(b.ymin for b in wbs)
        ymax = max(b.ymax for b in wbs)
        zmin = min(b.zmin for b in wbs)
        zmax = max(b.zmax for b in wbs)
        ux0 = int(math.floor(xmin / 5.0))
        ux1 = int(math.ceil(xmax / 5.0))
        uy0 = int(math.floor(ymin / 5.0))
        uy1 = int(math.ceil(ymax / 5.0))
        uz0 = int(math.floor(zmin / 6.0))
        uz1 = int(math.ceil(zmax / 6.0))
        if ux0 < 0 or uy0 < 0 or uz0 < 0 or ux1 > gx or uy1 > gy or uz1 > gz:
            return -1.0, 0, 0
        x0, x1, y0, y1, z0, z1 = ux0, ux1, uy0, uy1, uz0, uz1
        if x1 <= x0 or y1 <= y0 or z1 <= z0 or z0 >= gz6_nonempty_limit:
            return -1.0, 0, 0

        if len(wbs) == 1:
            new_m = ~occ6_live[x0:x1, y0:y1, z0:z1]
            if not np.any(new_m):
                return 0.0, 0, 0
            tgt_sl = occ6[x0:x1, y0:y1, z0:z1]
            d_in = int(np.count_nonzero(new_m & tgt_sl))
            d_ov = int(np.count_nonzero(new_m & (~tgt_sl)))
        else:
            loc = np.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=bool)
            for b in wbs:
                bx0 = max(x0, int(math.floor(b.xmin / 5.0))) - x0
                bx1 = min(x1, int(math.ceil(b.xmax / 5.0))) - x0
                by0 = max(y0, int(math.floor(b.ymin / 5.0))) - y0
                by1 = min(y1, int(math.ceil(b.ymax / 5.0))) - y0
                bz0 = max(z0, int(math.floor(b.zmin / 6.0))) - z0
                bz1 = min(z1, int(math.ceil(b.zmax / 6.0))) - z0
                if bx1 > bx0 and by1 > by0 and bz1 > bz0:
                    loc[bx0:bx1, by0:by1, bz0:bz1] = True
            new_m = loc & (~occ6_live[x0:x1, y0:y1, z0:z1])
            if not np.any(new_m):
                return 0.0, 0, 0
            tgt_sl = occ6[x0:x1, y0:y1, z0:z1]
            d_in = int(np.count_nonzero(new_m & tgt_sl))
            d_ov = int(np.count_nonzero(new_m & (~tgt_sl)))

        if enforce_min_precision and ((d_in * 3 < d_ov) if is_thin_regime else (d_in * 2 < d_ov)):
            return -1.0, d_in, d_ov
        if z1 > gz6_nonempty_limit:
            sky_ov = int(np.count_nonzero(new_m[:, :, max(0, gz6_nonempty_limit - z0) :]))
            if sky_ov * 2 > d_in:
                return -1.0, d_in, d_ov
        if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
            z_sk0 = max(0, eff_gz6_sky_limit - z0)
            sk_sl = new_m[:, :, z_sk0:]
            sk_tot = int(np.count_nonzero(sk_sl))
            if sk_tot >= 8 and int(np.count_nonzero(sk_sl & tgt_sl[:, :, z_sk0:])) * 3 < sk_tot:
                return -1.0, d_in, d_ov

        if (
            enforce_min_precision
            and inter_live < int(0.18 * mesh_sub_total)
            and float(d_in) <= 0.22 * float(d_ov)
        ):
            return -1.0, d_in, d_ov

        u_cur = float(max(1, mesh_sub_total + over_live))
        iou_cur = float(inter_live) / u_cur
        iou_new = float(inter_live + d_in) / float(max(1, mesh_sub_total + over_live + d_ov))
        return (iou_new - iou_cur), d_in, d_ov

    def _protrusion_clear_against_existing(pid_c: str, pose_c: LDUFrame) -> bool:
        """O(1) spatial-hash check that no unmated stud protrusion clashes with existing protrusions."""
        prim_p = get_brick_primitive(pid_c)
        for p_spec in prim_p.ports:
            if p_spec.local_protrusion_box is not None:
                w_pf = pose_c.compose(p_spec.local_frame)
                w_pr = p_spec.local_protrusion_box.transform(w_pf)
                for cell in SpatialBoxHashIndex.box_cells(w_pr):
                    for _, other_pr in state._protrusion_cells.get(cell, ()):
                        if w_pr.intersects(other_pr):
                            return False
        return True

    skin_lp_calls = 0
    max_skin_lp_calls = 36 if is_thin_regime else 22
    consec_skin_lp_fails = 0
    deferred_lp_in_batch = False
    batch_new_commits: List[Tuple[str, LDUFrame, bool]] = []

    def _try_commit_part(
        pid_c: str,
        pose_c: LDUFrame,
        grounded_c: bool,
        check_csg_strict: bool = False,
        budget_lp: bool = False,
        defer_lp: bool = False,
    ) -> bool:
        nonlocal base_witness, base_lp_eta, skin_lp_calls, consec_skin_lp_fails, deferred_lp_in_batch
        if rem_inv is not None and rem_inv.get(pid_c, 0) <= 0:
            return False
        prim_c = get_brick_primitive(pid_c)
        wbs = tuple(b.transform(pose_c) for b in prim_c.hull_boxes)
        if any(_box_hits_forbidden(wb, forbidden_support_boxes) for wb in wbs):
            return False
        if not check_monotone_insertion_corridor(state, pid_c, pose_c):
            return False
        if check_csg_strict and not _protrusion_clear_against_existing(pid_c, pose_c):
            return False

        tok = None
        try:
            tok = state.push_trial_part(pid_c, pose_c, is_grounded=grounded_c)
            if not grounded_c and len(tok.added_adj_other_insts) == 0:
                state.pop_trial_part(tok)
                return False
            trial_iid = tok.inst_id
            stable_ok = False
            if (
                grounded_c
                and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                and base_lp_eta <= 1.0 - 1e-5
            ):
                stable_ok = True
            elif foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE and base_lp_eta <= 1.0 - 1e-5:
                can_ext, peak_d = kernel.validator.can_extend_plumb_primal_witness(
                    state, trial_iid, plumb_deltas, base_witness=base_witness, hop_cache=plumb_hop_cache
                )
                if can_ext and max(base_lp_eta, peak_d) <= 1.0 - 1e-5:
                    stable_ok = True
            if (
                not stable_ok
                and defer_lp
                and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                and base_lp_eta <= 0.82
                and prim_c.mass_grams <= 8.0
            ):
                stable_ok = True
                deferred_lp_in_batch = True
            if not stable_ok:
                if budget_lp and (skin_lp_calls >= max_skin_lp_calls or consec_skin_lp_fails >= 10):
                    plumb_hop_cache.pop(trial_iid, None)
                    state.pop_trial_part(tok)
                    return False
                if budget_lp:
                    skin_lp_calls += 1
                eq_res = kernel.validator.evaluate(state, foundation_policy=foundation_policy)
                if not eq_res.is_stable:
                    if budget_lp:
                        consec_skin_lp_fails += 1
                    plumb_hop_cache.pop(trial_iid, None)
                    state.pop_trial_part(tok)
                    return False
                if budget_lp:
                    consec_skin_lp_fails = 0
                base_witness = eq_res.primal_witness
                base_lp_eta = float(eq_res.peak_utilization)
                plumb_deltas.clear()
            if plumb_hop_cache:
                w_zmax = max(b.zmax for b in wbs)
                for adj_id in tok.added_adj_other_insts:
                    if adj_id in plumb_hop_cache and min(b.zmin for b in state._parts[adj_id].world_boxes) >= w_zmax:
                        plumb_hop_cache.pop(adj_id, None)
            state.pop_trial_part(tok)
            tok = None

            state.place_root_part(pid_c, pose_c, is_grounded=grounded_c)
            build_mold_counts[pid_c] = build_mold_counts.get(pid_c, 0) + 1
            if rem_inv is not None:
                rem_inv[pid_c] = rem_inv.get(pid_c, 0) - 1
            _add_boxes_to_live_occ6(wbs)
            if defer_lp:
                batch_new_commits.append((pid_c, pose_c, grounded_c))
            return True
        except ValueError:
            if tok is not None:
                plumb_hop_cache.pop(tok.inst_id, None)
                state.pop_trial_part(tok)
            return False

    # Execute planned bottom-up core & jumper layers; queue cantilevered spans that need neighbor
    # bridging within the same Z layer for a second pass right after pier anchors commit!
    for z0 in sorted(planned_by_z.keys()):
        layer_items = planned_by_z[z0]
        is_gr = z0 == 0
        pending = list(layer_items)
        for _pass in range(4):
            if not pending:
                break
            still_pending: List[Tuple[str, LDUFrame]] = []
            progressed = False
            for pid_l, pose_l in pending:
                if _try_commit_part(pid_l, pose_l, grounded_c=is_gr, check_csg_strict=False, budget_lp=False):
                    progressed = True
                else:
                    still_pending.append((pid_l, pose_l))
            pending = still_pending
            if not progressed:
                break
        # Fallback: if any reserved multi-box slope/inverted-slope failed insertion/LP, fill its `core_occ` cells
        # with `1x1` bricks/plates so zero structural holes propagate to upper courses!
        if pending:
            for pid_f, pose_f in pending:
                prim_f = get_brick_primitive(pid_f)
                if len(prim_f.hull_boxes) <= 1:
                    continue
                wbs_f = (
                    [prim_f.hull_boxes[0].transform(pose_f)]
                    if pid_f in ("44728", "99780", "99207")
                    else [b.transform(pose_f) for b in prim_f.hull_boxes]
                )
                f_xmin = min(b.xmin for b in wbs_f)
                f_xmax = max(b.xmax for b in wbs_f)
                f_ymin = min(b.ymin for b in wbs_f)
                f_ymax = max(b.ymax for b in wbs_f)
                f_zmax = max(b.zmax for b in wbs_f)
                p_f0 = z0 // 8
                p_f1 = min(n_plates_z, f_zmax // 8)
                if p_f0 >= n_plates_z:
                    continue
                ph_fb = layer_phase[p_f0]
                rx0_f = max(0, (f_xmin - ph_fb[0]) // 20)
                rx1_f = min(w_x, (f_xmax - ph_fb[0]) // 20)
                ry0_f = max(0, (f_ymin - ph_fb[1]) // 20)
                ry1_f = min(w_y, (f_ymax - ph_fb[1]) // 20)
                if "3024" in kit_set:
                    p1_box = get_brick_primitive("3024").hull_boxes[0]
                    for pq_fb in range(p_f0, p_f1):
                        z_fb = pq_fb * 8
                        for fx_i in range(rx0_f, rx1_f):
                            for fy_i in range(ry0_f, ry1_f):
                                if core_occ_pre_sculpt[fx_i, fy_i, pq_fb]:
                                    pose_p1 = LDUFrame(
                                        0,
                                        (
                                            fx_i * 20 + ph_fb[0] - p1_box.xmin,
                                            fy_i * 20 + ph_fb[1] - p1_box.ymin,
                                            z_fb - p1_box.zmin,
                                        ),
                                    )
                                    _try_commit_part("3024", pose_p1, grounded_c=(z_fb == 0), check_csg_strict=False, budget_lp=False)

    # 4. Exact Global `ΔIoU` Open-Catalog Sculptural, Jumper & 24-Orientation SNOT Skinning
    #    Build skinning candidate part list from `kit_ids`:
    sculptural_molds: List[Tuple[str, str, bool]] = []
    snot_anchor_molds: List[str] = []
    snot_side_skin_molds: List[str] = []
    jumper_cap_molds: List[str] = []

    for pid in kit_ids:
        ent = CATALOG_RETRIEVAL_INDEX.get_entry(pid)
        cls_name = ent.taxonomy_class if ent is not None else "CORE_PLATE"
        in_comp = pid not in EXTENDED_BRICK_CATALOG
        if cls_name == "SNOT_ANCHOR":
            snot_anchor_molds.append(pid)
            if ent is not None and ent.studs_x * ent.studs_y <= 4:
                sculptural_molds.append((pid, cls_name, in_comp))
            if ent is not None and ent.studs_x * ent.studs_y <= 2:
                jumper_cap_molds.append(pid)
        elif cls_name == "JUMPER_HALF_STUD":
            continue
        elif cls_name in SCULPTURAL_TAXONOMY_CLASSES:
            if not is_thin_regime and pid == "6231":
                continue
            prim_sc = get_brick_primitive(pid)
            n_hbox = len(prim_sc.hull_boxes)
            all_coarse_20x24 = n_hbox >= 1 and all(
                (b.xmax - b.xmin) % 20 == 0 and (b.ymax - b.ymin) % 20 == 0 and (b.zmax - b.zmin) % 24 == 0
                for b in prim_sc.hull_boxes
            )
            # Exclude unsliced 1-box, >0.90-fill, or coarse 20x20x24 cuboids masquerading as slopes/arches/panels
            is_genuine_thin_wall = (
                cls_name == "THIN_PANEL"
                and n_hbox == 1
                and min(
                    prim_sc.hull_boxes[0].xmax - prim_sc.hull_boxes[0].xmin,
                    prim_sc.hull_boxes[0].ymax - prim_sc.hull_boxes[0].ymin,
                ) <= 12
            )
            if (
                n_hbox == 1
                and not is_genuine_thin_wall
                and (
                    cls_name in ("ARCH", "INVERTED_SLOPE", "CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE", "THIN_PANEL")
                    or (cls_name == "XY_WEDGE_ROUND" and ent is not None and ent.studs_x * ent.studs_y > 1)
                )
            ) or (
                ent is not None
                and (
                    (ent.fill_ratio > 0.90 and ent.studs_x * ent.studs_y > 1)
                    or (all_coarse_20x24 and cls_name in ("STEEP_SLOPE", "PLANAR_SLOPE", "CURVED_SLOPE"))
                    or (is_thin_regime and cls_name != "THIN_PANEL" and ent.height_ldu >= 32 and ent.fill_ratio > 0.60)
                )
            ):
                continue
            sculptural_molds.append((pid, cls_name, in_comp))
            if ent is not None and ent.height_ldu <= 16 and ent.studs_x * ent.studs_y <= 12:
                snot_side_skin_molds.append(pid)
            if (
                ent is not None
                and ent.studs_x * ent.studs_y <= (2 if is_thin_regime else 4)
                and (
                    ent.height_ldu <= 24
                    or (cls_name == "THIN_PANEL" and ent.height_ldu <= 72)
                )
            ):
                jumper_cap_molds.append(pid)
        elif cls_name == "SMOOTH_TILE" and ent is not None and ent.studs_x * ent.studs_y <= 12:
            snot_side_skin_molds.append(pid)
        elif cls_name == "CORE_PLATE" and ent is not None and ent.studs_x * ent.studs_y <= 12:
            if ent.studs_x * ent.studs_y <= (8 if is_thin_regime else 6):
                sculptural_molds.append((pid, cls_name, in_comp))
            snot_side_skin_molds.append(pid)
            if ent.studs_x * ent.studs_y <= 4:
                jumper_cap_molds.append(pid)
        elif cls_name == "CORE_BRICK" and ent is not None and ent.studs_x * ent.studs_y <= 6:
            sculptural_molds.append((pid, cls_name, in_comp))
            if pid in ("3005", "3004", "3622"):
                jumper_cap_molds.append(pid)

    def _cap_cls_rank(pid_j: str) -> int:
        ent_j = CATALOG_RETRIEVAL_INDEX.get_entry(pid_j)
        if ent_j is None:
            return 5
        c_j = ent_j.taxonomy_class
        if c_j == "THIN_PANEL":
            return 0
        if c_j in ("CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE", "XY_WEDGE_ROUND"):
            return 1
        if c_j == "CORE_PLATE":
            return 2
        return 3

    jumper_cap_molds.sort(key=lambda p_j: (_cap_cls_rank(p_j), p_j))
    snot_side_skin_molds.sort(
        key=lambda p_s: (
            0
            if (ent_s := CATALOG_RETRIEVAL_INDEX.get_entry(p_s)) is not None and ent_s.taxonomy_class == "SMOOTH_TILE"
            else (1 if ent_s is not None and ent_s.taxonomy_class == "CORE_PLATE" else 2),
            -(ent_s.studs_x * ent_s.studs_y if ent_s is not None else 1),
            p_s,
        )
    )

    # Helper: count uncovered mesh sub-voxels in tight local port neighborhood so we rank true boundary ports first
    def _port_uncovered_count(pos: Tuple[int, int, int], w_norm: Tuple[int, int, int]) -> int:
        gx, gy, gz = occ6.shape
        px, py, pz = pos
        r_xy = 22 if w_norm == (0, 0, 1) else 28
        x0 = max(0, (px - r_xy) // 5)
        x1 = min(gx, (px + r_xy) // 5 + 1)
        y0 = max(0, (py - r_xy) // 5)
        y1 = min(gy, (py + r_xy) // 5 + 1)
        z0 = max(0, (pz - (20 if w_norm[2] <= 0 else 0)) // 6)
        z1 = min(gz, (pz + (26 if w_norm == (0, 0, 1) else 32)) // 6 + 1)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return 0
        return int(np.count_nonzero(occ6[x0:x1, y0:y1, z0:z1] & (~occ6_live[x0:x1, y0:y1, z0:z1])))

    gx6, gy6, gz6 = occ6.shape
    mold_has_top_stud: Dict[str, bool] = {
        pid_m: any(
            pt_m.polarity == PortPolarity.STUD and pt_m.local_frame.z_axis == (0, 0, 1)
            for pt_m in get_brick_primitive(pid_m).ports
        )
        for pid_m in {*(m[0] for m in sculptural_molds), *jumper_cap_molds}
    }

    def _caps_uncovered_upper_mesh(
        wbs_chk: Sequence[LDUBox],
        cls_chk: str = "",
        pid_chk: str = "",
        pose_chk: Optional[LDUFrame] = None,
    ) -> bool:
        """Return True if placing a studless (or partially studless on thin frames) mold at `wbs_chk` caps uncovered upper mesh."""
        if is_thin_regime and cls_chk in ("THIN_PANEL", "SNOT_ANCHOR"):
            return False
        ux0 = max(0, min(b.xmin for b in wbs_chk) // 5)
        ux1 = min(gx6, int(math.ceil(max(b.xmax for b in wbs_chk) / 5.0)))
        uy0 = max(0, min(b.ymin for b in wbs_chk) // 5)
        uy1 = min(gy6, int(math.ceil(max(b.ymax for b in wbs_chk) / 5.0)))
        uz1 = min(gz6, int(math.ceil(max(b.zmax for b in wbs_chk) / 6.0)))
        uz2 = min(gz6, uz1 + 3)
        if uz2 <= uz1 or ux1 <= ux0 or uy1 <= uy0:
            return False
        if not mold_has_top_stud.get(pid_chk, False):
            above_uncov = int(
                np.count_nonzero(occ6[ux0:ux1, uy0:uy1, uz1:uz2] & (~occ6_live[ux0:ux1, uy0:uy1, uz1:uz2]))
            )
            return above_uncov >= (3 if is_thin_regime else 8)
        if is_thin_regime and pid_chk and pose_chk is not None and cls_chk in ("XY_WEDGE_ROUND", "CURVED_SLOPE", "PLANAR_SLOPE"):
            above_sl = (occ6[ux0:ux1, uy0:uy1, uz1:uz2] & (~occ6_live[ux0:ux1, uy0:uy1, uz1:uz2])).copy()
            if not above_sl.any():
                return False
            for pt_u in get_brick_primitive(pid_chk).ports:
                if pt_u.polarity == PortPolarity.STUD:
                    wf_u = pose_chk.compose(pt_u.local_frame)
                    if wf_u.z_axis == (0, 0, 1):
                        sx0_u = max(ux0, (wf_u.pos[0] - 10) // 5) - ux0
                        sx1_u = min(ux1, (wf_u.pos[0] + 10) // 5) - ux0
                        sy0_u = max(uy0, (wf_u.pos[1] - 10) // 5) - uy0
                        sy1_u = min(uy1, (wf_u.pos[1] + 10) // 5) - uy0
                        if sx1_u > sx0_u and sy1_u > sy0_u:
                            above_sl[sx0_u:sx1_u, sy0_u:sy1_u, :] = False
            return int(np.count_nonzero(above_sl)) >= 3
        return False

    # Grounded perimeter pass (`z = 0`): place grounded sculptural/plate/brick/SNOT molds on uncovered ground cells
    if foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE:
        ground_cands: List[Tuple[float, str, LDUFrame, bool]] = []
        seen_ground: Set[Tuple[str, int, Tuple[int, int, int]]] = set()
        ph0 = layer_phase[0]
        for ix in range(w_x):
            for iy in range(w_y):
                cx0 = (ix * 20 + ph0[0]) // 5
                cy0 = (iy * 20 + ph0[1]) // 5
                if cx0 + 4 > gx6 or cy0 + 4 > gy6:
                    continue
                if not np.any(occ6[cx0 : cx0 + 4, cy0 : cy0 + 4, :4] & (~occ6_live[cx0 : cx0 + 4, cy0 : cy0 + 4, :4])):
                    continue
                px_g = ix * 20 + ph0[0] + 10
                py_g = iy * 20 + ph0[1] + 10
                for pid_s, cls_s, is_comp_s in sculptural_molds:
                    for rid_t, dx_t, dy_t, dz_t, rboxes_t in _fast_upward_stud_templates(pid_s):
                        pos_g = (px_g + dx_t, py_g + dy_t, dz_t)
                        gkey = (pid_s, rid_t, pos_g)
                        if gkey in seen_ground:
                            continue
                        seen_ground.add(gkey)
                        d_iou = _eval_fast_pose_delta_iou(
                            pid_s, rid_t, pos_g[0], pos_g[1], pos_g[2], require_ground_z0=True
                        )
                        if d_iou <= 2.5e-5:
                            continue
                        wbs_t = tuple(
                            LDUBox(
                                b[0] + px_g,
                                b[1] + py_g,
                                b[2],
                                b[3] + px_g,
                                b[4] + py_g,
                                b[5],
                            )
                            for b in rboxes_t
                        )
                        if (not mold_has_top_stud.get(pid_s, True) or is_thin_regime) and _caps_uncovered_upper_mesh(
                            wbs_t, cls_s, pid_s, LDUFrame(rid_t, pos_g)
                        ):
                            continue
                        if state.check_boxes_collision(wbs_t):
                            continue
                        eps = (1.5e-5 if cls_s in SCULPTURAL_TAXONOMY_CLASSES else 0.0) - 1.5e-5 * float(
                            min(6, build_mold_counts.get(pid_s, 0))
                        ) + (
                            3.0e-5
                            if build_mold_counts.get(pid_s, 0) == 0
                            and cls_s in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "THIN_PANEL", "SNOT_ANCHOR")
                            else 0.0
                        )
                        ground_cands.append((d_iou + eps, pid_s, LDUFrame(rid_t, pos_g), is_comp_s))
        ground_cands.sort(key=lambda x: (-x[0], build_mold_counts.get(x[1], 0), x[1], x[2].rot_id, x[2].pos))
        for _, pid_g, pose_g, _ in ground_cands:
            d_now = _eval_fast_pose_delta_iou(
                pid_g, pose_g.rot_id, pose_g.pos[0], pose_g.pos[1], pose_g.pos[2], require_ground_z0=True
            )
            if d_now <= 2.5e-5:
                continue
            wbs_gn = tuple(b.transform(pose_g) for b in get_brick_primitive(pid_g).hull_boxes)
            if state.check_boxes_collision(wbs_gn):
                continue
            _try_commit_part(pid_g, pose_g, grounded_c=True, check_csg_strict=True, budget_lp=True)

    # Precompute upward-stud templates & overhang inverted-slope molds once per build
    mold_templates_s4 = [
        (pid_s, cls_s, is_comp_s, _fast_upward_stud_templates(pid_s))
        for pid_s, cls_s, is_comp_s in sculptural_molds
    ]
    overhang_inv_molds = [
        (pid_s, cls_s, is_comp_s)
        for pid_s, cls_s, is_comp_s in sculptural_molds
        if cls_s in ("INVERTED_SLOPE", "ARCH")
        or pid_s in ("3024", "3023", "3623", "3710", "3022", "24307", "24299")
    ]

    # Multi-wave frontier sculptural + jumper + SNOT skinning (`up to eff_max_waves fast waves; early-exits when 0 committed`)
    eff_max_waves = min(_max_waves, 2)
    for _wave in range(eff_max_waves):
        snap_parts_w = [(pl_s.part_id, pl_s.pose, pl_s.is_grounded) for _, pl_s in sorted(state.placed_parts.items())]
        snap_occ6_w = occ6_live.copy()
        snap_inter_w = inter_live
        snap_over_w = over_live
        snap_counts_w = dict(build_mold_counts)
        snap_rem_w = dict(rem_inv) if rem_inv is not None else None
        snap_witness_w = base_witness
        snap_eta_w = base_lp_eta
        snap_deltas_w = dict(plumb_deltas)
        deferred_lp_in_batch = False
        batch_new_commits = []

        frontier = state.get_active_frontier()
        ranked_ports: List[Tuple[int, object]] = []
        for ep in frontier.ports:
            if ep.status.is_exposed and ep.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
                uc_cnt = _port_uncovered_count(ep.world_frame.pos, ep.world_normal)
                if uc_cnt > 0:
                    ranked_ports.append((uc_cnt, ep))
        if not ranked_ports:
            break
        ranked_ports.sort(
            key=lambda item: (
                0 if item[1].world_normal[2] == 0 else 1,
                -item[0],
                item[1].world_frame.pos[2],
                item[1].world_frame.pos[0],
                item[1].world_frame.pos[1],
            )
        )
        port_cap = (56 if _wave == 0 else 28) if is_thin_regime else ((128 if mesh_sub_total >= 5000 else 80) if _wave == 0 else 48)
        active_ports = [item[1] for item in ranked_ports[:port_cap]]

        scored_single: List[Tuple[float, str, LDUFrame, bool]] = []
        seen_poses: Set[Tuple[str, int, Tuple[int, int, int]]] = set()

        for ep in active_ports:
            ux, uy, uz = ep.world_frame.pos
            u_rid = ep.world_frame.rot_id
            u_pol = ep.polarity
            u_ct = ep.connector_type
            wn = ep.world_normal

            if wn == (0, 0, 1) and u_pol == PortPolarity.STUD:
                for pid_s, cls_s, is_comp_s, tmpls_s in mold_templates_s4:
                    if rem_inv is not None and rem_inv.get(pid_s, 0) <= 0:
                        continue
                    for rid_t, dx_t, dy_t, dz_t, rboxes_t in tmpls_s:
                        pos_w = (ux + dx_t, uy + dy_t, uz + dz_t)
                        pkey = (pid_s, rid_t, pos_w)
                        if pkey in seen_poses:
                            continue
                        seen_poses.add(pkey)
                        d_iou = _eval_fast_pose_delta_iou(pid_s, rid_t, pos_w[0], pos_w[1], pos_w[2])
                        if d_iou <= 2.5e-5:
                            continue
                        wbs_t = tuple(
                            LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                            for b in rboxes_t
                        )
                        if _wave == 0 and (not mold_has_top_stud.get(pid_s, True) or is_thin_regime) and _caps_uncovered_upper_mesh(
                            wbs_t, cls_s, pid_s, LDUFrame(rid_t, pos_w)
                        ):
                            continue
                        if state.check_boxes_collision(wbs_t):
                            continue
                        eps = -2.0e-5 * float(min(10, build_mold_counts.get(pid_s, 0)))
                        if build_mold_counts.get(pid_s, 0) == 0 and cls_s in (
                            "CURVED_SLOPE",
                            "PLANAR_SLOPE",
                            "STEEP_SLOPE",
                            "XY_WEDGE_ROUND",
                            "THIN_PANEL",
                            "SNOT_ANCHOR",
                            "INVERTED_SLOPE",
                            "ARCH",
                            "SMOOTH_TILE",
                        ):
                            eps += 3.5e-5
                        if cls_s in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "INVERTED_SLOPE", "ARCH", "STEEP_SLOPE", "THIN_PANEL"):
                            eps += 2.0e-5
                        elif cls_s == "PLANAR_SLOPE":
                            eps += 1.5e-5
                        elif cls_s == "SMOOTH_TILE":
                            eps += 3.0e-5 if smooth_mode else 1.5e-5
                        scored_single.append((d_iou + eps, pid_s, LDUFrame(rid_t, pos_w), is_comp_s))
            elif u_pol == PortPolarity.ANTI_STUD and ep.world_frame.z_axis == (0, 0, 1) and uz >= 16:
                for pid_s, cls_s, is_comp_s in overhang_inv_molds:
                    if rem_inv is not None and rem_inv.get(pid_s, 0) <= 0:
                        continue
                    for tmpl in _get_anchor_templates(u_rid, u_pol, u_ct, pid_s):
                        rid_t, dx_t, dy_t, dz_t, rboxes_t, _, _, _ = tmpl
                        if LDUFrame(rid_t, (0, 0, 0)).z_axis != (0, 0, 1):
                            continue
                        pos_w = (ux + dx_t, uy + dy_t, uz + dz_t)
                        pkey = (pid_s, rid_t, pos_w)
                        if pkey in seen_poses:
                            continue
                        seen_poses.add(pkey)
                        d_iou = _eval_fast_pose_delta_iou(pid_s, rid_t, pos_w[0], pos_w[1], pos_w[2])
                        if d_iou <= 2.5e-5:
                            continue
                        wbs_t = tuple(
                            LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                            for b in rboxes_t
                        )
                        _, s_din, s_dov = _eval_boxes_delta_iou(wbs_t)
                        if s_din < s_dov:
                            continue
                        if state.check_boxes_collision(wbs_t):
                            continue
                        ov_first_b = 3.5e-5 if build_mold_counts.get(pid_s, 0) == 0 else 0.0
                        scored_single.append(
                            (
                                d_iou + 3.0e-5 + ov_first_b - 2.0e-5 * float(min(10, build_mold_counts.get(pid_s, 0))),
                                pid_s,
                                LDUFrame(rid_t, pos_w),
                                is_comp_s,
                            )
                        )
            elif wn[2] == 0 and u_pol == PortPolarity.STUD:
                for pid_s in snot_side_skin_molds:
                    if rem_inv is not None and rem_inv.get(pid_s, 0) <= 0:
                        continue
                    is_comp_s = pid_s not in EXTENDED_BRICK_CATALOG
                    for tmpl in _get_anchor_templates(u_rid, u_pol, u_ct, pid_s):
                        rid_t, dx_t, dy_t, dz_t, rboxes_t, _, _, _ = tmpl
                        pos_w = (ux + dx_t, uy + dy_t, uz + dz_t)
                        pkey = (pid_s, rid_t, pos_w)
                        if pkey in seen_poses:
                            continue
                        seen_poses.add(pkey)
                        d_iou = _eval_fast_pose_delta_iou(pid_s, rid_t, pos_w[0], pos_w[1], pos_w[2])
                        if d_iou <= 2.5e-5:
                            continue
                        wbs_t = tuple(
                            LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                            for b in rboxes_t
                        )
                        if state.check_boxes_collision(wbs_t):
                            continue
                        scored_single.append(
                            (
                                d_iou + 8.0e-4 - 4.0e-5 * float(min(10, build_mold_counts.get(pid_s, 0))),
                                pid_s,
                                LDUFrame(rid_t, pos_w),
                                is_comp_s,
                            )
                        )

        # Evaluate 2-part `(15573 jumper + half-stud cap)` & `(SNOT anchor + sideways tile/plate)` micro-motifs
        # IN THE SAME UNIFIED `ΔIoU` PRIORITY QUEUE as single parts so high-precision half-stud tier ledges
        # (`b4_unseen_01`, `b3_unseen_01`, `b2_unseen_02`, `b1_ext_e8`) win BEFORE low-precision smooth tiles cap studs!
        micro_pairs: List[Tuple[float, str, LDUFrame, str, LDUFrame, bool]] = []
        run_micro_wave = (_wave == 0) or (_wave == 1 and is_thin_regime)
        if run_micro_wave and (
            ("15573" in kit_set and jumper_cap_molds) or (snot_anchor_molds and snot_side_skin_molds)
        ):
            up_ports = [
                ep_u
                for ep_u in active_ports
                if ep_u.polarity == PortPolarity.STUD and ep_u.world_normal == (0, 0, 1)
            ][: (44 if mesh_sub_total >= 5000 else 28)]

            if _wave in (0, 1) and "15573" in kit_set and jumper_cap_molds:
                j_anc_cap = 28 if mesh_sub_total >= 2500 else 18
                j_mold_cap = 22 if (is_thin_regime or mesh_sub_total < 4500) else 14
                cand_j_anchors: List[Tuple[int, int, int, bool]] = [
                    (ep.world_frame.pos[0], ep.world_frame.pos[1], ep.world_frame.pos[2], False)
                    for ep in up_ports[:j_anc_cap]
                ]
                if _wave == 0 and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE and min(w_x, w_y) <= 2:
                    ph0_m = layer_phase[0]
                    for ix_g in range(w_x):
                        for iy_g in range(w_y):
                            cand_j_anchors.append((ix_g * 20 + ph0_m[0] + 10, iy_g * 20 + ph0_m[1] + 10, 0, True))
                for ux, uy, uz, is_g_j in cand_j_anchors:
                    for rid_j, dx_j, dy_j, dz_j, rboxes_j in _fast_upward_stud_templates("15573"):
                        pos_j = (ux + dx_j, uy + dy_j, uz + dz_j)
                        wbs_j = tuple(
                            LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                            for b in rboxes_j
                        )
                        if min(b.zmin for b in wbs_j) < 0 or state.check_boxes_collision(wbs_j):
                            continue
                        jf_w = LDUFrame(rid_j, pos_j)
                        s_ports = [
                            jf_w.compose(pt_s.local_frame)
                            for pt_s in get_brick_primitive("15573").ports
                            if pt_s.polarity == PortPolarity.STUD
                        ]
                        if not s_ports:
                            continue
                        wx_s, wy_s, wz_s = s_ports[0].pos
                        for cap_pid in jumper_cap_molds[:j_mold_cap]:
                            ent_cap = CATALOG_RETRIEVAL_INDEX.get_entry(cap_pid)
                            cls_cap = ent_cap.taxonomy_class if ent_cap is not None else "CORE_PLATE"
                            for rid_c, dx_c, dy_c, dz_c, rboxes_c in _fast_upward_stud_templates(cap_pid):
                                pos_c = (wx_s + dx_c, wy_s + dy_c, wz_s + dz_c)
                                if (
                                    _eval_fast_pose_delta_iou(
                                        cap_pid,
                                        rid_c,
                                        pos_c[0],
                                        pos_c[1],
                                        pos_c[2],
                                        enforce_min_precision=(not is_thin_regime),
                                    )
                                    <= 0.0
                                ):
                                    continue
                                wbs_c = tuple(
                                    LDUBox(b[0] + wx_s, b[1] + wy_s, b[2] + wz_s, b[3] + wx_s, b[4] + wy_s, b[5] + wz_s)
                                    for b in rboxes_c
                                )
                                if state.check_boxes_collision(wbs_c) or (
                                    (not mold_has_top_stud.get(cap_pid, True) or is_thin_regime)
                                    and _caps_uncovered_upper_mesh(wbs_j + wbs_c, cls_cap, cap_pid, LDUFrame(rid_c, pos_c))
                                ):
                                    continue
                                d_pair, _, _ = _eval_boxes_delta_iou(
                                    wbs_j + wbs_c, enforce_min_precision=(not is_thin_regime)
                                )
                                if d_pair > 1.5e-4:
                                    c_first_b = (
                                        3.5e-5
                                        if build_mold_counts.get(cap_pid, 0) == 0
                                        and cls_cap in ("CURVED_SLOPE", "XY_WEDGE_ROUND", "THIN_PANEL", "SNOT_ANCHOR")
                                        else 0.0
                                    )
                                    micro_pairs.append(
                                        (d_pair + c_first_b, "15573", jf_w, cap_pid, LDUFrame(rid_c, pos_c), is_g_j)
                                    )

            if snot_anchor_molds and snot_side_skin_molds and (_wave == 0 or (is_thin_regime and _wave <= 1)) and (is_thin_regime or mesh_sub_total < 4500):
                anc_cap_n = 4 if (is_thin_regime and mesh_sub_total < 900) else 3
                sk_cap_n = 8
                anc_try = [a for a in ("4070", "44728", "99780", "87087", "11211", "30414") if a in snot_anchor_molds][:anc_cap_n]
                cand_s_anchors: List[Tuple[int, int, int, bool]] = [
                    (ep.world_frame.pos[0], ep.world_frame.pos[1], ep.world_frame.pos[2], False)
                    for ep in up_ports[: (14 if is_thin_regime else 6)]
                ]
                if (
                    is_thin_regime
                    and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                    and min(w_x, w_y) <= 3
                ):
                    ph0_s = layer_phase[0]
                    for ix_gs in range(w_x):
                        for iy_gs in range(w_y):
                            cx0_g = (ix_gs * 20 + ph0_s[0]) // 5
                            cy0_g = (iy_gs * 20 + ph0_s[1]) // 5
                            if (
                                cx0_g + 4 <= gx6
                                and cy0_g + 4 <= gy6
                                and int(np.count_nonzero(occ6[cx0_g : cx0_g + 4, cy0_g : cy0_g + 4, 0])) >= 4
                                and (occ6[cx0_g : cx0_g + 4, cy0_g : cy0_g + 4, :6] & (~occ6_live[cx0_g : cx0_g + 4, cy0_g : cy0_g + 4, :6])).any()
                            ):
                                cand_s_anchors.append((ix_gs * 20 + ph0_s[0] + 10, iy_gs * 20 + ph0_s[1] + 10, 0, True))
                for ux, uy, uz, is_g_s in cand_s_anchors[: (16 if is_thin_regime else 8)]:
                    for anc_pid in anc_try:
                        prim_anc = get_brick_primitive(anc_pid)
                        for rid_a, dx_a, dy_a, dz_a, rboxes_a in _fast_upward_stud_templates(anc_pid):
                            pos_a = (ux + dx_a, uy + dy_a, uz + dz_a)
                            if is_g_s:
                                if min(b[2] for b in rboxes_a) != 0:
                                    continue
                                if _eval_fast_pose_delta_iou(anc_pid, rid_a, pos_a[0], pos_a[1], pos_a[2], require_ground_z0=True) <= 0.0:
                                    continue
                            wbs_a = tuple(
                                LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                                for b in rboxes_a
                            )
                            if min(b.zmin for b in wbs_a) < 0 or state.check_boxes_collision(wbs_a):
                                continue
                            af_w = LDUFrame(rid_a, pos_a)
                            lat_ports = [
                                (af_w.compose(pt_l.local_frame), pt_l.polarity, pt_l.connector_type)
                                for pt_l in prim_anc.ports
                                if pt_l.polarity == PortPolarity.STUD and af_w.compose(pt_l.local_frame).z_axis[2] == 0
                            ]
                            if not lat_ports:
                                continue
                            for lf_w, l_pol, l_ct in lat_ports[: (4 if is_thin_regime else 2)]:
                                wx_l, wy_l, wz_l = lf_w.pos
                                for sk_pid in snot_side_skin_molds[:sk_cap_n]:
                                    for tmpl_k in _get_anchor_templates(lf_w.rot_id, l_pol, l_ct, sk_pid):
                                        rid_k, dx_k, dy_k, dz_k, rboxes_k, _, _, _ = tmpl_k
                                        pos_k = (wx_l + dx_k, wy_l + dy_k, wz_l + dz_k)
                                        if _eval_fast_pose_delta_iou(sk_pid, rid_k, pos_k[0], pos_k[1], pos_k[2]) <= 1.0e-4:
                                            continue
                                        wbs_k = tuple(
                                            LDUBox(b[0] + wx_l, b[1] + wy_l, b[2] + wz_l, b[3] + wx_l, b[4] + wy_l, b[5] + wz_l)
                                            for b in rboxes_k
                                        )
                                        if min(b.zmin for b in wbs_k) < 0 or state.check_boxes_collision(wbs_k):
                                            continue
                                        if any(ba.intersects(bk) for ba in wbs_a for bk in wbs_k):
                                            continue
                                        d_pair, _, _ = _eval_boxes_delta_iou(
                                            wbs_a + wbs_k, enforce_min_precision=(not is_thin_regime)
                                        )
                                        if d_pair > 2.5e-4:
                                            anc_first_b = 2.2e-4 if build_mold_counts.get(anc_pid, 0) == 0 else 0.0
                                            micro_pairs.append(
                                                (d_pair + 1.5e-4 + anc_first_b, anc_pid, af_w, sk_pid, LDUFrame(rid_k, pos_k), is_g_s)
                                            )

        # Unified greedy `ΔIoU` commit queue over both 1-part and 2-part frontier actions
        unified_actions: List[Tuple[float, int, str, LDUFrame, Optional[str], Optional[LDUFrame], bool]] = [
            (sc_1, 1, p_1, f_1, None, None, False) for sc_1, p_1, f_1, _ in scored_single
        ]
        if micro_pairs:
            micro_pairs.sort(key=lambda x: (-x[0], x[1], x[2].rot_id, x[2].pos, x[3], x[4].rot_id, x[4].pos))
            anc_pair_cnt: Dict[Tuple[str, Tuple[int, int, int]], int] = {}
            added_mp = 0
            for sc_2, p1_m, f1_m, p2_m, f2_m, g1_m in micro_pairs:
                akey_m = (p1_m, f1_m.pos)
                c_ak = anc_pair_cnt.get(akey_m, 0)
                if c_ak >= 2:
                    continue
                anc_pair_cnt[akey_m] = c_ak + 1
                unified_actions.append((sc_2, 2, p1_m, f1_m, p2_m, f2_m, g1_m))
                added_mp += 1
                if added_mp >= 56:
                    break
        unified_actions.sort(
            key=lambda x: (-x[0], x[1], build_mold_counts.get(x[2], 0), x[2], x[3].rot_id, x[3].pos)
        )

        committed_in_wave = 0
        for _, kind_a, p1_a, f1_a, p2_a, f2_a, g1_a in unified_actions:
            if kind_a == 1:
                d_now = _eval_fast_pose_delta_iou(
                    p1_a, f1_a.rot_id, f1_a.pos[0], f1_a.pos[1], f1_a.pos[2]
                )
                if d_now <= 2.5e-5:
                    continue
                wbs_now = tuple(b.transform(f1_a) for b in get_brick_primitive(p1_a).hull_boxes)
                if state.check_boxes_collision(wbs_now):
                    continue
                if _try_commit_part(
                    p1_a, f1_a, grounded_c=False, check_csg_strict=True, budget_lp=True, defer_lp=True
                ):
                    committed_in_wave += 1
            elif p2_a is not None and f2_a is not None:
                wb1 = tuple(b.transform(f1_a) for b in get_brick_primitive(p1_a).hull_boxes)
                wb2 = tuple(b.transform(f2_a) for b in get_brick_primitive(p2_a).hull_boxes)
                if state.check_boxes_collision(wb1) or state.check_boxes_collision(wb2):
                    continue
                if not check_monotone_insertion_corridor(state, p2_a, f2_a) or not _protrusion_clear_against_existing(p2_a, f2_a):
                    continue
                dn_pair, _, _ = _eval_boxes_delta_iou(wb1 + wb2, enforce_min_precision=(not is_thin_regime))
                if dn_pair <= 1.2e-4:
                    continue
                if _try_commit_part(p1_a, f1_a, grounded_c=g1_a, check_csg_strict=True, budget_lp=True, defer_lp=True):
                    if _try_commit_part(p2_a, f2_a, grounded_c=False, check_csg_strict=True, budget_lp=True, defer_lp=True):
                        committed_in_wave += 2

        if deferred_lp_in_batch:
            skin_lp_calls += 1
            eq_wave = kernel.validator.evaluate(state, foundation_policy=foundation_policy)
            if eq_wave.is_stable:
                base_witness = eq_wave.primal_witness
                base_lp_eta = float(eq_wave.peak_utilization)
                plumb_deltas.clear()
            else:
                # Roll back speculative batch and replay per-part with exact individual LP certification
                state = AssemblyState()
                for p_rb, f_rb, g_rb in snap_parts_w:
                    state.place_root_part(p_rb, f_rb, is_grounded=g_rb)
                occ6_live[:] = snap_occ6_w
                inter_live = snap_inter_w
                over_live = snap_over_w
                build_mold_counts = snap_counts_w
                if rem_inv is not None and snap_rem_w is not None:
                    rem_inv.clear()
                    rem_inv.update(snap_rem_w)
                base_witness = snap_witness_w
                base_lp_eta = snap_eta_w
                plumb_deltas.clear()
                plumb_deltas.update(snap_deltas_w)
                plumb_hop_cache.clear()
                deferred_lp_in_batch = False
                committed_in_wave = 0
                for p_rp, f_rp, g_rp in batch_new_commits:
                    if _try_commit_part(
                        p_rp, f_rp, grounded_c=g_rp, check_csg_strict=True, budget_lp=True, defer_lp=False
                    ):
                        committed_in_wave += 1

        if committed_in_wave == 0:
            break

    # 4b. Exact Global `ΔIoU` Top-Surface & Leaf Part Prune (`~0.3 ms`): greedily prune any top-surface or
    #     degree-1 part that is net-negative at the final assembly IoU (up to 2 top-down passes), verifying
    #     monotone insertion corridors, full graph connectivity, and 6-DoF static stability on every removal.
    if len(state.placed_parts) > 1 and mesh_sub_total > 0:
        def _try_build_pruned_state(skip_iids: Set[int]) -> Optional[AssemblyState]:
            cand_st = AssemblyState()
            for iid_r in sorted(state.placed_parts.keys()):
                if iid_r in skip_iids:
                    continue
                pl_r = state.placed_parts[iid_r]
                if not check_monotone_insertion_corridor(cand_st, pl_r.part_id, pl_r.pose):
                    return None
                prev_m_len = len(cand_st.matched_port_pairs)
                try:
                    cand_st.place_root_part(pl_r.part_id, pl_r.pose, is_grounded=pl_r.is_grounded)
                except ValueError:
                    return None
                if not pl_r.is_grounded and len(cand_st.matched_port_pairs) == prev_m_len:
                    return None
            if not cand_st.placed_parts:
                return None
            eq_pr = kernel.validator.evaluate(cand_st, foundation_policy=foundation_policy)
            return cand_st if eq_pr.is_stable else None

        for _pr_pass in range(16):
            if len(state.placed_parts) <= 1:
                break
            adj_insts: Dict[int, Set[int]] = {k_i: set() for k_i in state.placed_parts}
            inst_zmax: Dict[int, int] = {
                k_i: max(b.zmax for b in pl_i.world_boxes) for k_i, pl_i in state.placed_parts.items()
            }
            has_higher_nbr: Set[int] = set()
            for (ia_m, _), (ib_m, _) in state.matched_port_pairs.items():
                if ia_m != ib_m and ia_m in adj_insts and ib_m in adj_insts:
                    adj_insts[ia_m].add(ib_m)
                    adj_insts[ib_m].add(ia_m)
                    if inst_zmax[ib_m] > inst_zmax[ia_m]:
                        has_higher_nbr.add(ia_m)
                    elif inst_zmax[ia_m] > inst_zmax[ib_m]:
                        has_higher_nbr.add(ib_m)
            leaf_cands = [
                iid_l
                for iid_l, pl_l in state.placed_parts.items()
                if (not pl_l.is_grounded and (len(adj_insts[iid_l]) == 1 or iid_l not in has_higher_nbr))
                or (pl_l.is_grounded and len(adj_insts[iid_l]) == 0)
            ]
            if not leaf_cands:
                break

            occ6_cnt = np.zeros(occ6.shape, dtype=np.uint16)
            inst_masks: Dict[int, Tuple[int, int, int, int, int, int, np.ndarray]] = {}
            for iid_p, pl_p in state.placed_parts.items():
                wbs_p = pl_p.world_boxes
                ux0 = max(0, int(math.floor(min(b.xmin for b in wbs_p) / 5.0)))
                ux1 = min(gx6, int(math.ceil(max(b.xmax for b in wbs_p) / 5.0)))
                uy0 = max(0, int(math.floor(min(b.ymin for b in wbs_p) / 5.0)))
                uy1 = min(gy6, int(math.ceil(max(b.ymax for b in wbs_p) / 5.0)))
                uz0 = max(0, int(math.floor(min(b.zmin for b in wbs_p) / 6.0)))
                uz1 = min(gz6, int(math.ceil(max(b.zmax for b in wbs_p) / 6.0)))
                if ux1 <= ux0 or uy1 <= uy0 or uz1 <= uz0:
                    continue
                loc_m = np.zeros((ux1 - ux0, uy1 - uy0, uz1 - uz0), dtype=bool)
                for b in wbs_p:
                    bx0 = max(ux0, int(math.floor(b.xmin / 5.0))) - ux0
                    bx1 = min(ux1, int(math.ceil(b.xmax / 5.0))) - ux0
                    by0 = max(uy0, int(math.floor(b.ymin / 5.0))) - uy0
                    by1 = min(uy1, int(math.ceil(b.ymax / 5.0))) - uy0
                    bz0 = max(uz0, int(math.floor(b.zmin / 6.0))) - uz0
                    bz1 = min(uz1, int(math.ceil(b.zmax / 6.0))) - uz0
                    if bx1 > bx0 and by1 > by0 and bz1 > bz0:
                        loc_m[bx0:bx1, by0:by1, bz0:bz1] = True
                occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] += loc_m.astype(np.uint16)
                inst_masks[iid_p] = (ux0, ux1, uy0, uy1, uz0, uz1, loc_m)

            neg_cands: List[Tuple[float, int]] = []
            iou_base = float(inter_live) / float(max(1, mesh_sub_total + over_live))
            for iid_l in leaf_cands:
                m_info = inst_masks.get(iid_l)
                if m_info is None:
                    continue
                ux0, ux1, uy0, uy1, uz0, uz1, loc_m = m_info
                uniq_m = loc_m & (occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] == 1)
                if not np.any(uniq_m):
                    continue
                tgt_u = occ6[ux0:ux1, uy0:uy1, uz0:uz1]
                lost_in = int(np.count_nonzero(uniq_m & tgt_u))
                saved_ov = int(np.count_nonzero(uniq_m & (~tgt_u)))
                if inter_live - lost_in <= 0:
                    continue
                iou_after = float(inter_live - lost_in) / float(max(1, mesh_sub_total + over_live - saved_ov))
                if iou_after > iou_base + 1.0e-5:
                    neg_cands.append((iou_after - iou_base, iid_l))

            if not neg_cands:
                break
            neg_cands.sort(key=lambda x: (-x[0], -x[1]))

            # Fast batch try first; if blocked by a bridge dependency, verify each candidate one-by-one!
            batch_ids = {iid_n for _, iid_n in neg_cands}
            batch_st = _try_build_pruned_state(batch_ids) if len(batch_ids) > 1 else None
            if batch_st is not None:
                occ6_new = np.zeros_like(occ6)
                for pl_k in batch_st.placed_parts.values():
                    for b in pl_k.world_boxes:
                        occ6_new[
                            max(0, b.xmin // 5) : min(gx6, (b.xmax + 4) // 5),
                            max(0, b.ymin // 5) : min(gy6, (b.ymax + 4) // 5),
                            max(0, b.zmin // 6) : min(gz6, (b.zmax + 5) // 6),
                        ] = True
                in_new = int(np.count_nonzero(occ6_new & occ6))
                ov_new = int(np.count_nonzero(occ6_new & (~occ6)))
                if in_new > 0 and (float(in_new) / float(max(1, mesh_sub_total + ov_new))) > iou_base + 1.0e-5:
                    if rem_inv is not None:
                        for iid_d in batch_ids:
                            pid_d = state.placed_parts[iid_d].part_id
                            rem_inv[pid_d] = rem_inv.get(pid_d, 0) + 1
                    state = batch_st
                    plumb_hop_cache.clear()
                    occ6_live = occ6_new
                    inter_live = in_new
                    over_live = ov_new
                    continue

            dropped_any = False
            for _, iid_one in neg_cands[:8]:
                if iid_one not in state.placed_parts:
                    continue
                m_info = inst_masks.get(iid_one)
                if m_info is None:
                    continue
                ux0, ux1, uy0, uy1, uz0, uz1, loc_m = m_info
                uniq_m = loc_m & (occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] == 1)
                if not np.any(uniq_m):
                    continue
                tgt_u = occ6[ux0:ux1, uy0:uy1, uz0:uz1]
                lost_in = int(np.count_nonzero(uniq_m & tgt_u))
                saved_ov = int(np.count_nonzero(uniq_m & (~tgt_u)))
                if inter_live - lost_in <= 0:
                    continue
                iou_cur_p = float(inter_live) / float(max(1, mesh_sub_total + over_live))
                iou_nxt_p = float(inter_live - lost_in) / float(max(1, mesh_sub_total + over_live - saved_ov))
                if iou_nxt_p <= iou_cur_p + 1.0e-5:
                    continue
                one_st = _try_build_pruned_state({iid_one})
                if one_st is not None:
                    pid_d = state.placed_parts[iid_one].part_id
                    if rem_inv is not None:
                        rem_inv[pid_d] = rem_inv.get(pid_d, 0) + 1
                    # Note: `place_root_part` re-indexes `inst_id` sequentially `0..N-1`; rebuild `occ6_cnt` on next pass
                    state = one_st
                    plumb_hop_cache.clear()
                    occ6_cnt[ux0:ux1, uy0:uy1, uz0:uz1] -= loc_m.astype(np.uint16)
                    inter_live -= lost_in
                    over_live -= saved_ov
                    dropped_any = True
                    break
            if not dropped_any:
                break
            # Sync `occ6_live` if single-item prune mutated `state` (`so any subsequent logic sees exact state`)
            occ6_live.fill(False)
            for pl_s in state.placed_parts.values():
                for b in pl_s.world_boxes:
                    occ6_live[
                        max(0, b.xmin // 5) : min(gx6, (b.xmax + 4) // 5),
                        max(0, b.ymin // 5) : min(gy6, (b.ymax + 4) // 5),
                        max(0, b.zmin // 6) : min(gz6, (b.zmax + 5) // 6),
                    ] = True

    # 5. Option 3B Zero-Cost Terminal Smooth-Tile Swap (`0.000 IoU` delta, `0` extra parts!)
    matched_now = state.matched_port_pairs
    for iid in sorted(state.placed_parts.keys()):
        pl = state.placed_parts[iid]
        tile_pid = PLATE_TO_SMOOTH_TILE_MAP.get(pl.part_id)
        if tile_pid is None:
            continue
        # In smooth_studless mode, allow all standard smooth tiles if license-valid; otherwise require in kit_set
        if not smooth_mode and tile_pid not in kit_set:
            continue
        if license_filter == "cc_by_4_0_only" and tile_pid not in EXTENDED_BRICK_CATALOG:
            ent_t = CATALOG_RETRIEVAL_INDEX.get_entry(tile_pid)
            if ent_t is None or ent_t.license_tier != "cc_by_4_0":
                continue
        if rem_inv is not None and rem_inv.get(tile_pid, 0) <= 0:
            continue
        prim_p = get_brick_primitive(pl.part_id)
        # Only swap upward-oriented horizontal plates (`zmax - zmin == 8`) with ZERO matched upward studs
        if (max(b.zmax for b in pl.world_boxes) - min(b.zmin for b in pl.world_boxes)) != 8:
            continue
        has_matched_top = any(
            (iid, p.port_id) in matched_now
            for p in prim_p.ports
            if p.polarity == PortPolarity.STUD
        )
        if has_matched_top:
            continue
        # Align `tile_pid` across `C4_Z_ROT_IDS` so LDraw-centered tiles (`e.g. 69729`) match `pl.world_boxes`
        prim_t = get_brick_primitive(tile_pid)
        if len(prim_t.hull_boxes) != 1 or len(pl.world_boxes) != 1:
            continue
        pb_w = pl.world_boxes[0]
        matched_tile_pose: Optional[LDUFrame] = None
        for rid_t in C4_Z_ROT_IDS:
            tb_l = prim_t.hull_boxes[0].transform(LDUFrame(rid_t, (0, 0, 0)))
            if (tb_l.xmax - tb_l.xmin) == (pb_w.xmax - pb_w.xmin) and (tb_l.ymax - tb_l.ymin) == (pb_w.ymax - pb_w.ymin):
                cand_tp = LDUFrame(rid_t, (pb_w.xmin - tb_l.xmin, pb_w.ymin - tb_l.ymin, pb_w.zmin - tb_l.zmin))
                if tuple(b.transform(cand_tp) for b in prim_t.hull_boxes) == pl.world_boxes:
                    matched_tile_pose = cand_tp
                    break
        if matched_tile_pose is None:
            continue
        try:
            state.reorient_part(iid, matched_tile_pose, new_part_id=tile_pid)
            plumb_hop_cache.pop(iid, None)
            if rem_inv is not None:
                rem_inv[pl.part_id] = rem_inv.get(pl.part_id, 0) + 1
                rem_inv[tile_pid] = rem_inv.get(tile_pid, 0) - 1
        except ValueError:
            continue

    if _eval_only:
        return (
            state,
            coarse_target_field,
            OpenCatalogRealizeDiagnostics(
                retrieval_result=ret_res,
                distinct_parts_placed=(),
                sculptural_parts_placed_count=0,
                jumper_parts_placed_count=0,
                snot_parts_placed_count=0,
                smooth_tiles_placed_count=0,
                exposed_upward_studs_count=0,
                mean_color_delta_e00=0.0,
            ),
        )

    # Evaluate upper-setback `15573` jumper phase-shift plans & compact thin-frame phase/threshold
    # candidates against exact `5x5x6 LDU` `occ6` IoU + 6-DoF static stability (`monotonically >= Plan 0`!)
    if (
        _forced_plan is None
        and inventory_counts is None
        and foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
        and mesh_sub_total > 0
    ):
        def _exact_certified_occ6_iou(
            st_chk: AssemblyState,
        ) -> Tuple[float, int, int, np.ndarray]:
            if not st_chk.placed_parts or not any(p.is_grounded for p in st_chk.placed_parts.values()):
                return -1.0, 0, 0, occ6_live
            # Verify single connected component across matched port pairs
            if len(st_chk.placed_parts) > 1:
                adj_c: Dict[int, Set[int]] = {k_c: set() for k_c in st_chk.placed_parts}
                for (ia_c, _), (ib_c, _) in st_chk.matched_port_pairs.items():
                    if ia_c != ib_c and ia_c in adj_c and ib_c in adj_c:
                        adj_c[ia_c].add(ib_c)
                        adj_c[ib_c].add(ia_c)
                roots_c = [k_c for k_c, p_c in st_chk.placed_parts.items() if p_c.is_grounded]
                vis_c: Set[int] = set(roots_c)
                dq_c = deque(roots_c)
                while dq_c:
                    cur_c = dq_c.popleft()
                    for nb_c in adj_c[cur_c]:
                        if nb_c not in vis_c:
                            vis_c.add(nb_c)
                            dq_c.append(nb_c)
                if len(vis_c) != len(st_chk.placed_parts):
                    return -1.0, 0, 0, occ6_live
            m_chk = np.zeros_like(occ6)
            for pl_chk in st_chk.placed_parts.values():
                for b_chk in pl_chk.world_boxes:
                    if _box_hits_forbidden(b_chk, forbidden_support_boxes):
                        return -1.0, 0, 0, occ6_live
                    ux0_c = int(math.floor(b_chk.xmin / 5.0))
                    ux1_c = int(math.ceil(b_chk.xmax / 5.0))
                    uy0_c = int(math.floor(b_chk.ymin / 5.0))
                    uy1_c = int(math.ceil(b_chk.ymax / 5.0))
                    uz0_c = int(math.floor(b_chk.zmin / 6.0))
                    uz1_c = int(math.ceil(b_chk.zmax / 6.0))
                    if (
                        ux0_c < 0
                        or uy0_c < 0
                        or uz0_c < 0
                        or ux1_c > gx6
                        or uy1_c > gy6
                        or uz1_c > gz6
                    ):
                        return -1.0, 0, 0, occ6_live
                    if ux1_c > ux0_c and uy1_c > uy0_c and uz1_c > uz0_c:
                        m_chk[ux0_c:ux1_c, uy0_c:uy1_c, uz0_c:uz1_c] = True
            if not kernel.validator.evaluate(st_chk, foundation_policy=foundation_policy).is_stable:
                return -1.0, 0, 0, occ6_live
            in_chk = int(np.count_nonzero(m_chk & occ6))
            ov_chk = int(np.count_nonzero(m_chk & (~occ6)))
            return float(in_chk) / float(max(1, mesh_sub_total + ov_chk)), in_chk, ov_chk, m_chk

        best_plan_iou, inter_0, over_0, mask_0 = _exact_certified_occ6_iou(state)
        if best_plan_iou >= 0.0:
            inter_live, over_live, occ6_live = inter_0, over_0, mask_0

        cand_forced_plans: List[
            Tuple[List[Tuple[int, int]], Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]], int]
        ] = []

        if "15573" in kit_set and not forbidden_support_boxes and n_plates_z >= (5 if is_thin_regime else 7) and mesh_sub_total < 15000:
            def _slice_align_sc(ph_chk: Tuple[int, int], p0_c: int, p1_c: int) -> float:
                cg_s = phase_grids[ph_chk][:, :, p0_c:p1_c]
                if is_thin_regime:
                    sol_s = cg_s >= 16
                    strad_s = (cg_s >= 6) & (cg_s < 16)
                    return float(np.sum(2 * cg_s[sol_s].astype(np.int32) - 24)) - 8.0 * float(
                        np.count_nonzero(strad_s)
                    )
                sol_s = cg_s >= 38
                strad_s = (cg_s >= 16) & (cg_s < 38)
                return float(np.sum(2 * cg_s[sol_s].astype(np.int32) - 64)) - 12.0 * float(
                    np.count_nonzero(strad_s)
                )

            jb_scored: List[
                Tuple[
                    float,
                    Tuple[int, int],
                    List[Tuple[int, int]],
                    Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
                ]
            ] = []
            for p_jb_c in range(1, n_plates_z - 2):
                for ph_up_c in ((10, 0), (0, 10), (10, 10)):
                    if ph_up_c == best_ground_phase:
                        continue
                    two_ax = (ph_up_c[0] != best_ground_phase[0]) and (ph_up_c[1] != best_ground_phase[1])
                    p_up0_c = p_jb_c + (2 if two_ax else 1)
                    if p_up0_c + 2 > n_plates_z:
                        continue
                    p_imm_end = min(n_plates_z, p_up0_c + 3)
                    imm_gain_c = _slice_align_sc(ph_up_c, p_up0_c, p_imm_end) - _slice_align_sc(
                        best_ground_phase, p_up0_c, p_imm_end
                    )
                    if imm_gain_c <= (8.0 if is_thin_regime else 20.0):
                        continue
                    up_gain_c = _slice_align_sc(ph_up_c, p_up0_c, n_plates_z) - _slice_align_sc(
                        best_ground_phase, p_up0_c, n_plates_z
                    )
                    if up_gain_c <= (24.0 if is_thin_regime else 80.0):
                        continue
                    jb_loss_c = (0.10 if is_thin_regime else 0.15) * max(
                        0.0, _slice_align_sc(best_ground_phase, p_jb_c, p_up0_c)
                    ) + (8.0 if is_thin_regime else 30.0) * (2.0 if two_ax else 1.0)
                    # Penalize placing a jumper bridge when the layer above the jumper is still wider on `best_ground_phase`
                    g_any_jb = int(
                        np.count_nonzero(
                            phase_grids[best_ground_phase][:, :, p_up0_c] >= (16 if is_thin_regime else 38)
                        )
                    )
                    u_sol_up = int(
                        np.count_nonzero(
                            phase_grids[ph_up_c][:, :, p_up0_c] >= (16 if is_thin_regime else 38)
                        )
                    )
                    if not is_thin_regime and g_any_jb >= u_sol_up + 4:
                        jb_loss_c += 35.0 * float(g_any_jb - u_sol_up)
                    gain_f_c = up_gain_c - jb_loss_c
                    if gain_f_c <= (10.0 if is_thin_regime else 30.0):
                        continue
                    l_ph_cand = [best_ground_phase] * p_up0_c + [ph_up_c] * (n_plates_z - p_up0_c)
                    if two_ax:
                        ph_mid_c = (ph_up_c[0], best_ground_phase[1])
                        jb_cand = {
                            p_jb_c: (best_ground_phase, ph_mid_c),
                            p_jb_c + 1: (ph_mid_c, ph_up_c),
                        }
                    else:
                        jb_cand = {p_jb_c: (best_ground_phase, ph_up_c)}
                    jb_scored.append((gain_f_c, ph_up_c, l_ph_cand, jb_cand))

            jb_scored.sort(key=lambda item: -item[0])
            if jb_scored:
                base_est_jb = _fast_est_plan_occ6_iou(layer_phase, {}, core_cell_thresh)
                jb_est_ranked: List[
                    Tuple[
                        float,
                        float,
                        Tuple[int, int],
                        List[Tuple[int, int]],
                        Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
                    ]
                ] = []
                seen_ph_est: Set[Tuple[int, int]] = set()
                top_jb_pool: List[
                    Tuple[
                        float,
                        Tuple[int, int],
                        List[Tuple[int, int]],
                        Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
                    ]
                ] = []
                for g_val, ph_u_s, l_ph_s, jb_s in jb_scored:
                    if ph_u_s not in seen_ph_est:
                        seen_ph_est.add(ph_u_s)
                        top_jb_pool.append((g_val, ph_u_s, l_ph_s, jb_s))
                for g_val, ph_u_s, l_ph_s, jb_s in jb_scored:
                    if len(top_jb_pool) >= 6:
                        break
                    if all(jb_s != ex[3] for ex in top_jb_pool):
                        top_jb_pool.append((g_val, ph_u_s, l_ph_s, jb_s))
                for g_val, ph_u_s, l_ph_s, jb_s in top_jb_pool:
                    est_j = _fast_est_plan_occ6_iou(l_ph_s, jb_s, core_cell_thresh)
                    if est_j >= base_est_jb - (0.030 if is_thin_regime else 0.020):
                        jb_est_ranked.append((est_j, g_val, ph_u_s, l_ph_s, jb_s))
                jb_est_ranked.sort(key=lambda item: (-item[0], -item[1]))
                max_jb_try = 2 if mesh_sub_total < 1200 else (1 if mesh_sub_total < 6000 else 0)
                for _, _, _, l_ph_s, jb_s in jb_est_ranked[:max_jb_try]:
                    cand_forced_plans.append((l_ph_s, jb_s, core_cell_thresh))

        if forbidden_support_boxes and not is_thin_regime and mesh_sub_total < 4500 and best_plan_iou < 0.905:
            cand_forced_plans.append(([best_ground_phase] * n_plates_z, {}, 46))

        if not is_thin_regime and not forbidden_support_boxes and 2800 <= mesh_sub_total < 25000 and best_plan_iou < 0.9245:
            thr_hi = (
                42
                if (
                    6000 <= mesh_sub_total < 12000
                    and best_plan_iou < 0.916
                    and _fast_est_plan_occ6_iou([best_ground_phase] * n_plates_z, {}, 42)
                    >= _fast_est_plan_occ6_iou([best_ground_phase] * n_plates_z, {}, 38) - 0.004
                )
                else 38
            )
            cand_forced_plans.append(([best_ground_phase] * n_plates_z, {}, thr_hi))

        if (
            not is_thin_regime
            and not forbidden_support_boxes
            and 15000 <= mesh_sub_total < 26000
            and best_plan_iou < 0.931
            and best_ground_phase != (0, 10)
            and _fast_est_plan_occ6_iou([(0, 10)] * n_plates_z, {}, 28)
            >= _fast_est_plan_occ6_iou(layer_phase, {}, core_cell_thresh) - 0.015
        ):
            cand_forced_plans.append(([(0, 10)] * n_plates_z, {}, 28))

        if mesh_sub_total < 1600 and not forbidden_support_boxes:
            alt_small_specs: Tuple[Tuple[Tuple[int, int], int], ...] = (
                (
                    ((0, 0), 8),
                    ((0, 10), 12),
                    ((10, 0), 14),
                    ((0, 10), 14),
                    ((0, 0), 18),
                )
                if is_legged_frame
                else (
                    (
                        ((0, 0), 14),
                        ((0, 0), 24),
                        ((0, 10), 8),
                        ((10, 0), 8),
                        ((10, 0), 14),
                        ((0, 10), 14),
                        ((0, 0), 18),
                    )
                    if (is_thin_regime and mesh_sub_total < 900)
                    else (
                        ((10, 10), 28),
                        ((10, 0), 14),
                        ((0, 10), 14),
                        ((0, 0), 18),
                    )
                )
            )
            max_small_try = 2 if is_legged_frame else 1
            ranked_alt_specs = sorted(
                (
                    (_fast_est_plan_occ6_iou([ph_alt] * n_plates_z, {}, thr_alt), ph_alt, thr_alt)
                    for ph_alt, thr_alt in alt_small_specs
                    if not (ph_alt == best_ground_phase and thr_alt == core_cell_thresh)
                ),
                key=lambda item: -item[0],
            )
            for _, ph_alt, thr_alt in ranked_alt_specs[:max_small_try]:
                cand_forced_plans.append(([ph_alt] * n_plates_z, {}, thr_alt))

        # Fast `< 4 ms` Single-Leg Horizontal-Slab Portfolio Candidate for elevated thin-tabletop meshes
        won_single_leg_slab = False
        if (
            is_thin_regime
            and not forbidden_support_boxes
            and mesh_sub_total < 1200
            and best_plan_iou < 0.55
            and "3005" in kit_set
            and "3024" in kit_set
            and "3023" in kit_set
        ):
            plates_rect_sl = [
                p_r
                for p_r in ("3795", "3020", "3021", "3022", "3666", "3710", "3623", "3023", "3024")
                if p_r in kit_set
            ]
            best_p_slab = -1
            best_slab_tp = 0
            for p_sl in range(4, n_plates_z):
                uz0_sl = (8 * p_sl) // 6
                uz1_sl = min(gz6, (8 * (p_sl + 1) + 5) // 6)
                tp_sl = int(np.count_nonzero(occ6[:, :, uz0_sl:uz1_sl]))
                if tp_sl > best_slab_tp and tp_sl >= int(0.42 * mesh_sub_total):
                    best_slab_tp = tp_sl
                    best_p_slab = p_sl
            if best_p_slab >= 4 and plates_rect_sl:
                uz0_sl = (8 * best_p_slab) // 6
                uz1_sl = min(gz6, (8 * (best_p_slab + 1) + 5) // 6)
                z_sl = 8 * best_p_slab
                for ox_sl, oy_sl in ((0, 0), (10, 0), (0, 10), (10, 10)):
                    slab_cells: Set[Tuple[int, int]] = set()
                    for ix_sl in range((gx6 * 5 - ox_sl) // 20):
                        for iy_sl in range((gy6 * 5 - oy_sl) // 20):
                            cx_sl = (ix_sl * 20 + ox_sl) // 5
                            cy_sl = (iy_sl * 20 + oy_sl) // 5
                            if int(np.count_nonzero(occ6[cx_sl : cx_sl + 4, cy_sl : cy_sl + 4, uz0_sl:uz1_sl])) >= 11:
                                slab_cells.add((ix_sl, iy_sl))
                    if len(slab_cells) < 6:
                        continue
                    xs_sl = [c_s[0] for c_s in slab_cells]
                    ys_sl = [c_s[1] for c_s in slab_cells]
                    leg_cands_sl: List[Tuple[int, int, int]] = []
                    for lx_s, ly_s in slab_cells:
                        cx_sl = (lx_s * 20 + ox_sl) // 5
                        cy_sl = (ly_s * 20 + oy_sl) // 5
                        if int(np.count_nonzero(occ6[cx_sl : cx_sl + 4, cy_sl : cy_sl + 4, 0])) > 0:
                            leg_cands_sl.append(
                                (int(np.count_nonzero(occ6[cx_sl : cx_sl + 4, cy_sl : cy_sl + 4, :uz0_sl])), lx_s, ly_s)
                            )
                    leg_cands_sl.sort(reverse=True)
                    for _, lx_s, ly_s in leg_cands_sl[:2]:
                        st_sl = AssemblyState()
                        px_sl = lx_s * 20 + ox_sl + 10
                        py_sl = ly_s * 20 + oy_sl + 10
                        zc_sl = 0
                        try:
                            while zc_sl + 24 <= z_sl:
                                st_sl.place_root_part("3005", LDUFrame(0, (px_sl, py_sl, zc_sl)), is_grounded=(zc_sl == 0))
                                zc_sl += 24
                            while zc_sl + 8 <= z_sl:
                                st_sl.place_root_part("3024", LDUFrame(0, (px_sl, py_sl, zc_sl)), is_grounded=(zc_sl == 0))
                                zc_sl += 8
                            rem_sl = set(slab_cells)
                            cov_sl: Set[Tuple[int, int]] = set()
                            first_p_sl = None
                            for pid_r in plates_rect_sl:
                                prim_r = get_brick_primitive(pid_r)
                                for rid_r in (0, 9):
                                    b0_r = prim_r.hull_boxes[0].transform(LDUFrame(rid_r, (0, 0, 0)))
                                    sx_r = (b0_r.xmax - b0_r.xmin) // 20
                                    sy_r = (b0_r.ymax - b0_r.ymin) // 20
                                    for ix0_r in range(min(xs_sl), max(xs_sl) - sx_r + 2):
                                        for iy0_r in range(min(ys_sl), max(ys_sl) - sy_r + 2):
                                            cs_r = {(ix0_r + dx_r, iy0_r + dy_r) for dx_r in range(sx_r) for dy_r in range(sy_r)}
                                            if (lx_s, ly_s) in cs_r and cs_r <= rem_sl:
                                                if first_p_sl is None or len(cs_r) > first_p_sl[0]:
                                                    first_p_sl = (
                                                        len(cs_r),
                                                        pid_r,
                                                        rid_r,
                                                        ix0_r * 20 + ox_sl - b0_r.xmin,
                                                        iy0_r * 20 + oy_sl - b0_r.ymin,
                                                        cs_r,
                                                    )
                            if first_p_sl is None:
                                continue
                            _, pid_r0, rid_r0, wx_r0, wy_r0, cs_r0 = first_p_sl
                            st_sl.place_root_part(pid_r0, LDUFrame(rid_r0, (wx_r0, wy_r0, z_sl)), is_grounded=False)
                            rem_sl -= cs_r0
                            cov_sl |= cs_r0
                            for _ in range(5):
                                if not rem_sl:
                                    break
                                best_ext_sl = None
                                for pid_e in plates_rect_sl:
                                    prim_e = get_brick_primitive(pid_e)
                                    for rid_e in (0, 9):
                                        b0_e = prim_e.hull_boxes[0].transform(LDUFrame(rid_e, (0, 0, 0)))
                                        sx_e = (b0_e.xmax - b0_e.xmin) // 20
                                        sy_e = (b0_e.ymax - b0_e.ymin) // 20
                                        for ix0_e in range(min(xs_sl), max(xs_sl) - sx_e + 2):
                                            for iy0_e in range(min(ys_sl), max(ys_sl) - sy_e + 2):
                                                cs_e = {(ix0_e + dx_e, iy0_e + dy_e) for dx_e in range(sx_e) for dy_e in range(sy_e)}
                                                if cs_e <= rem_sl:
                                                    tie_pose_sl = None
                                                    best_tie_tp = -1
                                                    for ce_s in cs_e:
                                                        for ct_s in cov_sl:
                                                            if abs(ce_s[0] - ct_s[0]) + abs(ce_s[1] - ct_s[1]) == 1 and ct_s != (lx_s, ly_s):
                                                                rid_u = 9 if abs(ce_s[0] - ct_s[0]) == 1 else 0
                                                                b0_u = get_brick_primitive("3023").hull_boxes[0].transform(LDUFrame(rid_u, (0, 0, 0)))
                                                                ixu = min(ce_s[0], ct_s[0])
                                                                iyu = min(ce_s[1], ct_s[1])
                                                                wbu = LDUBox(
                                                                    ixu * 20 + ox_sl,
                                                                    iyu * 20 + oy_sl,
                                                                    z_sl - 8,
                                                                    (max(ce_s[0], ct_s[0]) + 1) * 20 + ox_sl,
                                                                    (max(ce_s[1], ct_s[1]) + 1) * 20 + oy_sl,
                                                                    z_sl,
                                                                )
                                                                if not st_sl.check_boxes_collision((wbu,)):
                                                                    ttp = int(
                                                                        np.count_nonzero(
                                                                            occ6[
                                                                                wbu.xmin // 5 : (wbu.xmax + 4) // 5,
                                                                                wbu.ymin // 5 : (wbu.ymax + 4) // 5,
                                                                                wbu.zmin // 6 : (wbu.zmax + 5) // 6,
                                                                            ]
                                                                        )
                                                                    )
                                                                    if ttp > best_tie_tp:
                                                                        best_tie_tp = ttp
                                                                        tie_pose_sl = LDUFrame(
                                                                            rid_u,
                                                                            (ixu * 20 + ox_sl - b0_u.xmin, iyu * 20 + oy_sl - b0_u.ymin, z_sl - 8),
                                                                        )
                                                    if tie_pose_sl is not None:
                                                        sc_e = (len(cs_e), best_tie_tp)
                                                        if best_ext_sl is None or sc_e > best_ext_sl[0]:
                                                            best_ext_sl = (
                                                                sc_e,
                                                                pid_e,
                                                                rid_e,
                                                                ix0_e * 20 + ox_sl - b0_e.xmin,
                                                                iy0_e * 20 + oy_sl - b0_e.ymin,
                                                                cs_e,
                                                                tie_pose_sl,
                                                            )
                                if best_ext_sl is None:
                                    break
                                _, pid_e0, rid_e0, wx_e0, wy_e0, cs_e0, tie_pose_0 = best_ext_sl
                                st_sl.place_root_part("3023", tie_pose_0, is_grounded=False)
                                st_sl.place_root_part(pid_e0, LDUFrame(rid_e0, (wx_e0, wy_e0, z_sl)), is_grounded=False)
                                rem_sl -= cs_e0
                                cov_sl |= cs_e0
                        except ValueError:
                            continue
                        iou_sl, in_sl, ov_sl, mask_sl = _exact_certified_occ6_iou(st_sl)
                        if iou_sl > best_plan_iou + 1.0e-4:
                            best_plan_iou = iou_sl
                            state, inter_live, over_live, occ6_live = st_sl, in_sl, ov_sl, mask_sl
                            base_witness = None
                            base_lp_eta = 0.0
                            plumb_deltas.clear()
                            plumb_hop_cache.clear()
                            won_single_leg_slab = True

        for f_plan_try in (() if won_single_leg_slab else cand_forced_plans):
            st_try, _, _ = realize_open_catalog_sculpted(
                kernel=kernel,
                coarse_target_field=coarse_target_field,
                adaptive_target=adaptive_target,
                allowed_part_ids=allowed_part_ids,
                part_retrieval=part_retrieval,
                retrieval_top_k=retrieval_top_k,
                license_filter=license_filter,
                semantic_style_hint=semantic_style_hint,
                vertex_colors_rgb=None,
                verts_ldu=verts_ldu,
                inventory_counts=None,
                support_policy=support_policy,
                foundation_policy=foundation_policy,
                forbidden_support_boxes=forbidden_support_boxes,
                _forced_plan=f_plan_try,
                _pre_ret=ret_res,
                _eval_only=True,
                _max_waves=(2 if mesh_sub_total < 2200 else 1),
            )
            iou_try, in_try, ov_try, mask_try = _exact_certified_occ6_iou(st_try)
            if iou_try > best_plan_iou + 1.0e-4:
                best_plan_iou = iou_try
                state, inter_live, over_live, occ6_live = st_try, in_try, ov_try, mask_try
                base_witness = None
                base_lp_eta = 0.0
                plumb_deltas.clear()
                plumb_hop_cache.clear()

    if not _eval_only and rem_inv is None:
        st_port = refine_open_catalog_blind_portfolio(
            kernel=kernel,
            state=state,
            occ6=ad_occ6,
            kit_ids=tuple(kit_ids),
            forbidden_support_boxes=forbidden_support_boxes,
            foundation_policy=foundation_policy,
        )
        if st_port is not state:
            iou_pf, in_pf, ov_pf, mask_pf = _exact_certified_occ6_iou(st_port)
            if iou_pf >= 0.0:
                state, inter_live, over_live, occ6_live = st_port, in_pf, ov_pf, mask_pf
                base_witness = None
                base_lp_eta = 0.0
                plumb_deltas.clear()
                plumb_hop_cache.clear()
        if smooth_mode:
            _apply_terminal_tile_swap(state, kit_set)
            plumb_hop_cache.clear()
        else:
            present_molds = {p.part_id for p in state.placed_parts.values()}
            matched_now_t = state.matched_port_pairs
            for iid_t in sorted(state.placed_parts.keys()):
                pl_t = state.placed_parts[iid_t]
                tl_cand = PLATE_TO_SMOOTH_TILE_MAP.get(pl_t.part_id)
                if tl_cand is None or tl_cand not in kit_set or tl_cand in present_molds or len(pl_t.world_boxes) != 1:
                    continue
                prim_pt = get_brick_primitive(pl_t.part_id)
                if any((iid_t, pt.port_id) in matched_now_t for pt in prim_pt.ports if pt.polarity == PortPolarity.STUD):
                    continue
                _apply_terminal_tile_swap(state, {tl_cand}, target_iids=(iid_t,))
                if state.placed_parts[iid_t].part_id == tl_cand:
                    present_molds.add(tl_cand)

    # (Step 5b) Zero-IoU-delta inline decomposition of completely top-exposed, 100%-supported `24 LDU`
    #           rectangular bricks into `(8 LDU plate + 8 LDU plate + 8 LDU smooth tile)` of identical footprint
    #           (all eligible roof bricks in `smooth_studless`; at most 1 brick per unused tile mold in default mode).
    if not _eval_only and rem_inv is None and (smooth_mode or len(state.placed_parts) <= 90):
        matched_cur = state.matched_port_pairs
        present_molds_b = {p.part_id for p in state.placed_parts.values()}
        decomp_targets: Dict[int, Tuple[List[Tuple[str, LDUFrame]], bool]] = {}
        b_3024_loc = get_brick_primitive("3024").hull_boxes[0]
        b_3070b_loc = get_brick_primitive("3070b").hull_boxes[0]
        for iid_b in sorted(state.placed_parts.keys()):
            pl_b = state.placed_parts[iid_b]
            pt_pair = BRICK_TO_PLATE_AND_TILE_MAP.get(pl_b.part_id)
            if pt_pair is None or len(pl_b.world_boxes) != 1:
                continue
            wb_b = pl_b.world_boxes[0]
            if (wb_b.zmax - wb_b.zmin) != 24:
                continue
            pl_pid, tl_pid = pt_pair
            if not smooth_mode:
                if pl_pid not in kit_set or tl_pid not in kit_set or tl_pid in present_molds_b:
                    continue
            if license_filter == "cc_by_4_0_only":
                ok_lic = True
                for chk_pid in (pl_pid, tl_pid, "3024", "3070b"):
                    if chk_pid not in EXTENDED_BRICK_CATALOG:
                        ent_chk = CATALOG_RETRIEVAL_INDEX.get_entry(chk_pid)
                        if ent_chk is None or ent_chk.license_tier != "cc_by_4_0":
                            ok_lic = False
                if not ok_lic:
                    continue
            prim_bk = get_brick_primitive(pl_b.part_id)
            up_stud_ports = [
                p_bk
                for p_bk in prim_bk.ports
                if p_bk.polarity == PortPolarity.STUD and pl_b.pose.compose(p_bk.local_frame).z_axis == (0, 0, 1)
            ]
            if not up_stud_ports:
                continue
            n_mated_up = sum(1 for p_bk in up_stud_ports if (iid_b, p_bk.port_id) in matched_cur)
            if not smooth_mode and n_mated_up > 0:
                continue
            if smooth_mode and n_mated_up >= len(up_stud_ports):
                continue
            if not pl_b.is_grounded:
                n_anti = sum(1 for p_bk in prim_bk.ports if p_bk.polarity == PortPolarity.ANTI_STUD)
                n_mated_anti = sum(
                    1
                    for p_bk in prim_bk.ports
                    if p_bk.polarity == PortPolarity.ANTI_STUD and (iid_b, p_bk.port_id) in matched_cur
                )
                if n_anti == 0 or n_mated_anti < n_anti:
                    continue
            # Match orientation `(0 vs 9)` of `pl_pid` and `tl_pid` so `world_boxes` cover `wb_b` bit-for-bit
            found_seq: Optional[List[Tuple[str, LDUFrame]]] = None
            for rid_pl in (0, 9):
                b_pl0 = get_brick_primitive(pl_pid).hull_boxes[0].transform(LDUFrame(rid_pl, (0, 0, 0)))
                if (b_pl0.xmax - b_pl0.xmin) == (wb_b.xmax - wb_b.xmin) and (b_pl0.ymax - b_pl0.ymin) == (
                    wb_b.ymax - wb_b.ymin
                ):
                    f_p0 = LDUFrame(rid_pl, (wb_b.xmin - b_pl0.xmin, wb_b.ymin - b_pl0.ymin, wb_b.zmin - b_pl0.zmin))
                    f_p1 = LDUFrame(
                        rid_pl, (wb_b.xmin - b_pl0.xmin, wb_b.ymin - b_pl0.ymin, wb_b.zmin + 8 - b_pl0.zmin)
                    )
                    if n_mated_up == 0:
                        for rid_tl in (0, 9):
                            b_tl0 = get_brick_primitive(tl_pid).hull_boxes[0].transform(LDUFrame(rid_tl, (0, 0, 0)))
                            if (b_tl0.xmax - b_tl0.xmin) == (wb_b.xmax - wb_b.xmin) and (
                                b_tl0.ymax - b_tl0.ymin
                            ) == (wb_b.ymax - wb_b.ymin):
                                f_t2 = LDUFrame(
                                    rid_tl,
                                    (wb_b.xmin - b_tl0.xmin, wb_b.ymin - b_tl0.ymin, wb_b.zmin + 16 - b_tl0.zmin),
                                )
                                found_seq = [(pl_pid, f_p0), (pl_pid, f_p1), (tl_pid, f_t2)]
                                break
                    else:
                        seq_p: List[Tuple[str, LDUFrame]] = [(pl_pid, f_p0), (pl_pid, f_p1)]
                        for p_bk in sorted(up_stud_ports, key=lambda ps: pl_b.pose.compose(ps.local_frame).pos):
                            sx_w, sy_w, _ = pl_b.pose.compose(p_bk.local_frame).pos
                            if (iid_b, p_bk.port_id) in matched_cur:
                                seq_p.append(
                                    (
                                        "3024",
                                        LDUFrame(
                                            0,
                                            (
                                                sx_w - 10 - b_3024_loc.xmin,
                                                sy_w - 10 - b_3024_loc.ymin,
                                                wb_b.zmin + 16 - b_3024_loc.zmin,
                                            ),
                                        ),
                                    )
                                )
                            else:
                                seq_p.append(
                                    (
                                        "3070b",
                                        LDUFrame(
                                            0,
                                            (
                                                sx_w - 10 - b_3070b_loc.xmin,
                                                sy_w - 10 - b_3070b_loc.ymin,
                                                wb_b.zmin + 16 - b_3070b_loc.zmin,
                                            ),
                                        ),
                                    )
                                )
                        found_seq = seq_p
                    break
            if found_seq is not None:
                decomp_targets[iid_b] = (found_seq, pl_b.is_grounded)
                present_molds_b.add(tl_pid)

        if decomp_targets:
            trial_sm = AssemblyState()
            replay_ok = True
            for iid_orig in sorted(state.placed_parts.keys()):
                if iid_orig in decomp_targets:
                    seq_d, is_gr_d = decomp_targets[iid_orig]
                    try:
                        for idx_sd, (pid_sd, f_sd) in enumerate(seq_d):
                            if not check_monotone_insertion_corridor(trial_sm, pid_sd, f_sd):
                                replay_ok = False
                                break
                            trial_sm.place_root_part(pid_sd, f_sd, is_grounded=(is_gr_d and idx_sd == 0))
                        if not replay_ok:
                            break
                    except ValueError:
                        replay_ok = False
                        break
                else:
                    po = state.placed_parts[iid_orig]
                    try:
                        if not check_monotone_insertion_corridor(trial_sm, po.part_id, po.pose):
                            replay_ok = False
                            break
                        trial_sm.place_root_part(po.part_id, po.pose, is_grounded=po.is_grounded)
                    except ValueError:
                        replay_ok = False
                        break
            if replay_ok and kernel.validator.evaluate(trial_sm, foundation_policy=foundation_policy).is_stable:
                state = trial_sm
                base_witness = None
                base_lp_eta = 0.0
                plumb_deltas.clear()
                plumb_hop_cache.clear()

    if smooth_mode:
        tile_cap_ids = [t for t in ("87079", "3068b", "2431", "63864", "3069b", "3070b") if (t in kit_set or inventory_counts is None)]
        fr_sm = state.get_active_frontier()
        exp_up = [
            ep
            for ep in fr_sm.ports
            if ep.status.is_exposed
            and ep.polarity == PortPolarity.STUD
            and ep.world_normal == (0, 0, 1)
            and ep.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
        ]
        iou_before_cap = float(inter_live) / float(max(1, mesh_sub_total + over_live))
        for ep in exp_up:
            ux, uy, uz = ep.world_frame.pos
            # Skip if already matched in an earlier multi-stud tile placement
            if ep.global_port_id in state.matched_port_pairs:
                continue
            for t_pid in tile_cap_ids:
                placed_t = False
                for tmpl in _get_anchor_templates(ep.world_frame.rot_id, ep.polarity, ep.connector_type, t_pid):
                    rid_t, dx_t, dy_t, dz_t, rboxes_t, rports_t, _, _ = tmpl
                    pos_t = (ux + dx_t, uy + dy_t, uz + dz_t)
                    wbs_t = tuple(
                        LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                        for b in rboxes_t
                    )
                    if state.check_boxes_collision(wbs_t):
                        continue
                    # Ensure ALL anti-studs of `t_pid` mate with exposed upward studs (so it doesn't overhang in midair!)
                    all_anti_mated = True
                    for _, _, apx, apy, apz, apol, _ in rports_t:
                        if apol == PortPolarity.ANTI_STUD:
                            w_apos = (ux + apx, uy + apy, uz + apz)
                            if not any(
                                ep2.world_frame.pos == w_apos and ep2.global_port_id not in state.matched_port_pairs
                                for ep2 in exp_up
                            ):
                                all_anti_mated = False
                                break
                    if not all_anti_mated:
                        continue
                    d_iou_t, d_in_t, d_ov_t = _eval_boxes_delta_iou(wbs_t, enforce_min_precision=False)
                    proj_iou = float(inter_live + d_in_t) / float(max(1, mesh_sub_total + over_live + d_ov_t))
                    if proj_iou >= iou_before_cap - 0.0020:
                        if _try_commit_part(t_pid, LDUFrame(rid_t, pos_t), grounded_c=False, check_csg_strict=True):
                            placed_t = True
                            break
                if placed_t:
                    break

    # 6. Optional Perceptual CIE Lab (`CIEDE2000`) Chromatic Mapping
    mean_de00 = 0.0
    if vertex_colors_rgb is not None:
        v_pts_color = verts_ldu if verts_ldu is not None else adaptive_target.mesh_surf_pts
        mean_de00 = assign_perceptual_ldraw_colors(state, v_pts_color, vertex_colors_rgb)

    # Compute diagnostics & exact `5x5x6 LDU` target field
    fine_tf = _build_fine_target_field_from_occ6(occ6)
    distinct_placed = tuple(sorted({p.part_id for p in state.placed_parts.values()}))
    sculpt_cnt = 0
    jump_cnt = 0
    snot_cnt = 0
    tile_cnt = 0
    for p in state.placed_parts.values():
        ent = CATALOG_RETRIEVAL_INDEX.get_entry(p.part_id)
        cls_p = ent.taxonomy_class if ent is not None else ""
        if cls_p in SCULPTURAL_TAXONOMY_CLASSES:
            sculpt_cnt += 1
        if cls_p == "JUMPER_HALF_STUD":
            jump_cnt += 1
        if cls_p == "SNOT_ANCHOR" or LDUFrame(p.pose.rot_id, (0, 0, 0)).z_axis[2] == 0:
            snot_cnt += 1
        if cls_p == "SMOOTH_TILE" or p.part_id in PLATE_TO_SMOOTH_TILE_MAP.values():
            tile_cnt += 1

    diag = OpenCatalogRealizeDiagnostics(
        retrieval_result=ret_res,
        distinct_parts_placed=distinct_placed,
        sculptural_parts_placed_count=sculpt_cnt,
        jumper_parts_placed_count=jump_cnt,
        snot_parts_placed_count=snot_cnt,
        smooth_tiles_placed_count=tile_cnt,
        exposed_upward_studs_count=count_exposed_upward_studs(state),
        mean_color_delta_e00=round(mean_de00, 4),
    )
    return state, fine_tf, diag
