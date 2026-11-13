"""
Phase 10 (`P10-WS1`): Open-Catalog Canonical Mold Deduplication & Vectorized `uint64`
3D Shape/Semantic Retrieval Index (`16,085` Combined Parts -> Canonical Base Molds).

Provides:
  1. `pack_occ_5x5x6_to_u64(occ_5x5x6)`: Vectorized O(1)-per-cell packing of `5x5x6 LDU`
     boolean sub-voxel grids (`4x4x4 = 64` sub-voxels per `20x20x24 LDU` brick cell) into
     `uint64` bitmasks (`bit = sx + 4*sy + 16*sz`).
  2. `CatalogMoldEntry` & `CatalogRetrievalResult`: Provenance-, license-, and taxonomy-tagged
     canonical mechanical base molds deduplicated across all `16,085` parts in
     `COMBINED_BRICK_CATALOG` (`EXTENDED_BRICK_CATALOG` + `COMPILED_LDRAW_CATALOG`).
  3. `CatalogShapeRetrievalIndex` + `CATALOG_RETRIEVAL_INDEX` (`compiled_mold_retrieval_index.npz`):
     `< 3 ms` single-core online `top-K` retrieval of curved slopes, planar/double/corner slopes,
     steep slopes, inverted slopes, arches, XY diagonal wings/round elements, thin panels,
     `10 LDU` half-stud jumpers, `24`-orientation SNOT anchors, and studless tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    BrickPrimitive,
    ConnectorType,
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.ldraw_shadow_catalog_compiler import (
    COMBINED_BRICK_CATALOG,
    COMPILED_LDRAW_CATALOG,
)
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame

DEFAULT_RETRIEVAL_INDEX_NPZ: Path = Path(__file__).with_name("compiled_mold_retrieval_index.npz")

MOLD_TAXONOMY_CLASSES: Tuple[str, ...] = (
    "CORE_BRICK",
    "CORE_PLATE",
    "CURVED_SLOPE",
    "PLANAR_SLOPE",
    "STEEP_SLOPE",
    "INVERTED_SLOPE",
    "XY_WEDGE_ROUND",
    "ARCH",
    "THIN_PANEL",
    "JUMPER_HALF_STUD",
    "SNOT_ANCHOR",
    "SMOOTH_TILE",
)

SCULPTURAL_TAXONOMY_CLASSES: Tuple[str, ...] = (
    "CURVED_SLOPE",
    "PLANAR_SLOPE",
    "STEEP_SLOPE",
    "INVERTED_SLOPE",
    "XY_WEDGE_ROUND",
    "ARCH",
    "THIN_PANEL",
    "JUMPER_HALF_STUD",
    "SNOT_ANCHOR",
    "SMOOTH_TILE",
)

# Precomputed 64-bit weights for `(4, 4, 4)` sub-voxel packing: `bit = sx + 4*sy + 16*sz`
_BIT_SHIFTS_4X4X4: np.ndarray = (
    np.arange(4, dtype=np.uint64)[:, None, None]
    + (np.arange(4, dtype=np.uint64)[None, :, None] << np.uint64(2))
    + (np.arange(4, dtype=np.uint64)[None, None, :] << np.uint64(4))
)
_BIT_WEIGHTS_4X4X4: np.ndarray = np.uint64(1) << _BIT_SHIFTS_4X4X4

# Byte-wise popcount lookup table for vectorized `uint64` popcount in NumPy
_POPCOUNT_U8: np.ndarray = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

_PRINT_SUFFIX_RE = re.compile(r"^([0-9]{2,6}[a-z]?)(?:p[a-z0-9]+|d[0-9]+|c[0-9]+)$")

_NON_SYSTEM_KEYWORDS: Tuple[str, ...] = (
    "pattern",
    "sticker",
    "minifig",
    "duplo",
    "fabuland",
    "belville",
    "scala",
    "bionicle",
    "znap",
    "clikits",
    "primo",
    "quatro",
    "electric",
    "motor",
    "train",
    "wheel",
    "tyre",
    "tire",
    "windscreen",
    "mudguard",
    "baseplate",
    "RFID",
    "rfid",
    "friends",
    "animal",
    "plant",
    "weapon",
    "sword",
    "helmet",
    "hair",
    "head",
    "torso",
    "leg",
    "arm",
    "hand",
    "hips",
)


def popcount_u64_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized popcount over an arbitrary-shaped `np.uint64` array, returning `int32` counts."""
    u64 = np.ascontiguousarray(arr, dtype=np.uint64)
    if u64.size == 0:
        return np.zeros(u64.shape, dtype=np.int32)
    u8 = u64.view(np.uint8).reshape(*u64.shape, 8)
    return np.sum(_POPCOUNT_U8[u8], axis=-1, dtype=np.int32)


def pack_occ_5x5x6_to_u64(occ_5x5x6: np.ndarray) -> np.ndarray:
    """
    Pack a `(gx, gy, gz)` boolean `5x5x6 LDU` occupancy grid into a `(nx_c, ny_c, nz_c)` array
    of `np.uint64` brick-cell bitmasks (`nx_c = ceil(gx/4)`, etc.), zero-padding partial tail cells.
    """
    occ = np.asarray(occ_5x5x6, dtype=bool)
    gx, gy, gz = occ.shape
    nx_c = max(1, (gx + 3) // 4)
    ny_c = max(1, (gy + 3) // 4)
    nz_c = max(1, (gz + 3) // 4)
    pad_x = 4 * nx_c - gx
    pad_y = 4 * ny_c - gy
    pad_z = 4 * nz_c - gz
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        occ = np.pad(occ, ((0, pad_x), (0, pad_y), (0, pad_z)), mode="constant", constant_values=False)
    reshaped = occ.reshape(nx_c, 4, ny_c, 4, nz_c, 4).transpose(0, 2, 4, 1, 3, 5)
    weighted = np.where(reshaped, _BIT_WEIGHTS_4X4X4[None, None, None, :, :, :], np.uint64(0))
    return np.bitwise_or.reduce(weighted.reshape(nx_c, ny_c, nz_c, 64), axis=-1)


def rasterize_boxes_to_u64_cells(
    boxes: Sequence[LDUBox],
) -> Tuple[int, int, int, np.ndarray, np.ndarray]:
    """
    Rasterize origin-shifted world `LDUBox` cuboids (`xmin >= 0, ymin >= 0, zmin >= 0`) onto the
    `5x5x6 LDU` sub-voxel grid (matching `_rasterize_boxes_5x5x6` bit-for-bit) and return:
      `(sx_cells, sy_cells, sz_cells, cell_u64_grid, normal_6d_float32)`.
    """
    if not boxes:
        return 1, 1, 1, np.zeros((1, 1, 1), dtype=np.uint64), np.zeros(6, dtype=np.float32)
    xmax = max(b.xmax for b in boxes)
    ymax = max(b.ymax for b in boxes)
    zmax = max(b.zmax for b in boxes)
    sx = max(1, int(math.ceil(xmax / 20.0)))
    sy = max(1, int(math.ceil(ymax / 20.0)))
    sz = max(1, int(math.ceil(zmax / 24.0)))
    occ = np.zeros((4 * sx, 4 * sy, 4 * sz), dtype=bool)
    for b in boxes:
        x0 = max(0, int(math.floor(b.xmin / 5.0)))
        x1 = min(4 * sx, int(math.ceil(b.xmax / 5.0)))
        y0 = max(0, int(math.floor(b.ymin / 5.0)))
        y1 = min(4 * sy, int(math.ceil(b.ymax / 5.0)))
        z0 = max(0, int(math.floor(b.zmin / 6.0)))
        z1 = min(4 * sz, int(math.ceil(b.zmax / 6.0)))
        if x1 > x0 and y1 > y0 and z1 > z0:
            occ[x0:x1, y0:y1, z0:z1] = True

    u64_grid = pack_occ_5x5x6_to_u64(occ)

    # Compute exposed 6-axis surface area histogram in sub-voxels
    pad = np.pad(occ, 1, mode="constant", constant_values=False)
    c = pad[1:-1, 1:-1, 1:-1]
    n6 = np.array(
        [
            float(np.count_nonzero(c & ~pad[2:, 1:-1, 1:-1])),   # +X
            float(np.count_nonzero(c & ~pad[:-2, 1:-1, 1:-1])),  # -X
            float(np.count_nonzero(c & ~pad[1:-1, 2:, 1:-1])),   # +Y
            float(np.count_nonzero(c & ~pad[1:-1, :-2, 1:-1])),  # -Y
            float(np.count_nonzero(c & ~pad[1:-1, 1:-1, 2:])),   # +Z
            float(np.count_nonzero(c & ~pad[1:-1, 1:-1, :-2])),  # -Z
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(n6))
    if norm > 1e-6:
        n6 /= norm
    return sx, sy, sz, u64_grid, n6


def _canonical_mechanical_signature(prim: BrickPrimitive) -> Tuple[Any, ...]:
    """Translation-normalized mechanical `(hull_boxes, ports)` equivalence key for mold deduplication."""
    if not prim.hull_boxes:
        return ("empty", len(prim.ports))
    ox = min(b.xmin for b in prim.hull_boxes)
    oy = min(b.ymin for b in prim.hull_boxes)
    oz = min(b.zmin for b in prim.hull_boxes)
    bsig = tuple(
        sorted(
            (b.xmin - ox, b.ymin - oy, b.zmin - oz, b.xmax - ox, b.ymax - oy, b.zmax - oz)
            for b in prim.hull_boxes
        )
    )
    psig = tuple(
        sorted(
            (
                int(p.polarity),
                int(p.connector_type),
                p.local_frame.z_axis,
                p.local_frame.pos[0] - ox,
                p.local_frame.pos[1] - oy,
                p.local_frame.pos[2] - oz,
            )
            for p in prim.ports
        )
    )
    return (bsig, psig)


_PREFERRED_CANONICAL_IDS: Set[str] = {
    "3001", "3002", "3003", "3004", "3005", "3008", "3009", "3010", "2456", "3622",
    "3020", "3021", "3022", "3023", "3024", "3460", "3623", "3666", "3710", "3795", "3034",
    "3070b", "3069b", "63864", "2431", "6636", "4162", "3068b", "26603", "87079", "1751", "69729",
    "15573", "87580", "3794b", "11477", "15068", "93273", "6091", "88930", "85970", "85984", "54200",
    "3037", "3039", "3040b", "3040a", "3038", "3041", "3043", "3044c", "3046",
    "3298", "3299", "3300", "30363", "60477", "4286",
    "60481", "3678b", "3684c", "3685", "4460b", "2449",
    "3660", "3665", "3747b", "24201", "3659", "6182", "3307", "6005", "92950",
    "24299", "24307", "41769", "41770", "43722", "43723", "43710", "43711",
    "2420", "14719", "2357", "3062b", "6141", "60474",
    "4865a", "4864", "87552", "2362b", "4215b", "59349", "87544",
    "87087", "11211", "4733", "4070", "99780", "99207", "44728", "99206",
}


def _mold_id_preference_key(part_id: str, title: str) -> Tuple[int, int, int, int, str]:
    """Prefer canonical System mold IDs, curated `EXTENDED_BRICK_CATALOG` IDs, and unpatterned molds."""
    is_pref = 0 if part_id in _PREFERRED_CANONICAL_IDS else 1
    in_ext = 0 if part_id in EXTENDED_BRICK_CATALOG else 1
    tl = title.lower()
    is_pat = 1 if ("pattern" in tl or "sticker" in tl or "grille" in tl or _PRINT_SUFFIX_RE.match(part_id)) else 0
    return (is_pref, in_ext, is_pat, len(part_id), part_id)


def _classify_mold_and_hygiene(
    part_id: str,
    title: str,
    prim: BrickPrimitive,
    license_tier: str,
) -> Tuple[str, bool, bool]:
    """
    Return `(taxonomy_class, is_structural_buildable, is_flush_system_mold)` for a canonical mold.
    """
    tl = title.lstrip("~").strip().lower()
    if not prim.hull_boxes or not prim.ports:
        return ("CORE_BRICK", False, False)

    # Explicit curated special connectors (`jumper`, `SNOT`)
    if part_id in ("15573", "87580"):
        return ("JUMPER_HALF_STUD", True, True)
    if part_id in ("87087", "11211", "4733", "4070", "99780", "99207", "44728", "99206"):
        return ("SNOT_ANCHOR", True, True)

    if any(kw in tl for kw in _NON_SYSTEM_KEYWORDS):
        return ("CORE_BRICK", False, False)

    # Allow passive hollow-stud/tube auxiliary ports (`BAR_3_18MM`, `SMOOTH_PIN`), rejecting
    # Technic pins/axles/hinges/ball-joints/turntables/wheels or external protrusions in XY.
    _allowed_aux = (
        ConnectorType.STANDARD_STUD,
        ConnectorType.HOLLOW_STUD,
        ConnectorType.BAR_3_18MM,
        ConnectorType.SMOOTH_PIN,
    )
    if any(p.connector_type not in _allowed_aux for p in prim.ports):
        return ("CORE_BRICK", False, False)

    anti_z = [
        p
        for p in prim.ports
        if p.polarity == PortPolarity.ANTI_STUD
        and p.local_frame.z_axis == (0, 0, 1)
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    ]
    stud_z = [
        p
        for p in prim.ports
        if p.polarity == PortPolarity.STUD
        and p.local_frame.z_axis == (0, 0, 1)
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    ]
    snot_ports = [
        p
        for p in prim.ports
        if p.local_frame.z_axis[2] == 0
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    ]
    if not anti_z:
        return ("CORE_BRICK", False, False)

    bx0 = min(b.xmin for b in prim.hull_boxes)
    bx1 = max(b.xmax for b in prim.hull_boxes)
    by0 = min(b.ymin for b in prim.hull_boxes)
    by1 = max(b.ymax for b in prim.hull_boxes)
    bz0 = min(b.zmin for b in prim.hull_boxes)
    bz1 = max(b.zmax for b in prim.hull_boxes)
    dx = bx1 - bx0
    dy = by1 - by0
    dz = bz1 - bz0

    if dx <= 0 or dy <= 0 or dx > 160 or dy > 160 or dz not in (8, 16, 24, 32, 48, 72):
        return ("CORE_BRICK", False, False)
    if any(p.local_frame.pos[2] != bz0 for p in anti_z):
        return ("CORE_BRICK", False, False)
    if stud_z and any(p.local_frame.pos[2] != bz1 for p in stud_z):
        return ("CORE_BRICK", False, False)
    # Reject any auxiliary bar/pin port protruding outside `[bx0..bx1, by0..by1, bz0..bz1]`
    for p_aux in prim.ports:
        if p_aux.connector_type not in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
            if p_aux.local_protrusion_box is not None:
                return ("CORE_BRICK", False, False)
            ax, ay, az = p_aux.local_frame.pos
            if not (bx0 <= ax <= bx1 and by0 <= ay <= by1 and bz0 <= az <= bz1 + 4):
                return ("CORE_BRICK", False, False)

    # Check flush 20-LDU System stud-grid alignment (allowing auxiliary center-tube anti-studs!)
    if snot_ports and part_id not in EXTENDED_BRICK_CATALOG:
        return ("SNOT_ANCHOR", False, False)

    if dx % 20 != 0 or dy % 20 != 0:
        return ("CORE_BRICK", False, False)
    if part_id not in EXTENDED_BRICK_CATALOG and any(
        b.xmin % 5 != 0 or b.xmax % 5 != 0 or b.ymin % 5 != 0 or b.ymax % 5 != 0
        for b in prim.hull_boxes
    ):
        return ("CORE_BRICK", False, False)

    rem_x = bx0 % 20
    rem_y = by0 % 20
    aligned_anti_z = [
        p
        for p in anti_z
        if (p.local_frame.pos[0] - 10 - rem_x) % 20 == 0
        and (p.local_frame.pos[1] - 10 - rem_y) % 20 == 0
    ]
    if not aligned_anti_z:
        return ("CORE_BRICK", False, False)
    anti_mod_x = {(rem_x + 10) % 20}
    anti_mod_y = {(rem_y + 10) % 20}

    if (
        dz == 8
        and ("jumper" in tl or "stud in center" in tl or part_id in ("15573", "87580", "3794b"))
        and stud_z
        and (
            any((p.local_frame.pos[0] % 20) not in anti_mod_x for p in stud_z)
            or any((p.local_frame.pos[1] % 20) not in anti_mod_y for p in stud_z)
        )
    ):
        return ("JUMPER_HALF_STUD", True, True)

    if snot_ports:
        return ("SNOT_ANCHOR", True, True)

    tot_vol = sum(b.volume_ldu3 for b in prim.hull_boxes)
    env_vol = dx * dy * dz
    fill_ratio = float(tot_vol) / float(max(env_vol, 1))
    nx_s = dx // 20
    ny_s = dy // 20
    is_full_box = len(prim.hull_boxes) == 1 and tot_vol == env_vol

    # Rectangular tiles, plates, bricks
    if is_full_box and dz == 8 and len(stud_z) == 0 and len(anti_z) == nx_s * ny_s:
        if "round" in tl:
            return ("XY_WEDGE_ROUND", True, True)
        return ("SMOOTH_TILE", True, True)
    if is_full_box and dz == 8 and len(stud_z) == nx_s * ny_s and len(anti_z) == nx_s * ny_s:
        if "round" in tl:
            return ("XY_WEDGE_ROUND", True, True)
        return ("CORE_PLATE", True, True)
    if is_full_box and dz == 24 and len(stud_z) == nx_s * ny_s and len(anti_z) == nx_s * ny_s:
        if "round" in tl:
            return ("XY_WEDGE_ROUND", True, True)
        return ("CORE_BRICK", True, True)

    # Sculptural non-full-box System molds
    if "arch" in tl or "curved bottom" in tl or part_id == "3659":
        return ("ARCH", True, True)
    if "panel" in tl or "wall" in tl:
        return ("THIN_PANEL", True, True)
    if "inverted" in tl or "slope inv" in tl or part_id in ("3660", "3665"):
        return ("INVERTED_SLOPE", True, True)
    if any(k in tl for k in ("wing", "triangular", "without corner", "cut ", "facet", "corner round")):
        if "curved top" in tl or "dome" in tl:
            return ("CURVED_SLOPE", True, True)
        return ("XY_WEDGE_ROUND", True, True)
    if tl.startswith("tile") and fill_ratio < 0.96:
        if any(k in tl for k in ("round", "corner", "cut", "triangular", "quarter", "circle")):
            return ("XY_WEDGE_ROUND", True, True)
        return ("SMOOTH_TILE", True, True)
    if tl.startswith("tile") and dz == 8 and len(stud_z) == 0:
        return ("SMOOTH_TILE", True, True)
    if any(k in tl for k in ("curved", "bow", "dome", "rounded")):
        return ("CURVED_SLOPE", True, True)
    if any(k in tl for k in ("65", "75", "slope brick 50")) or (dz >= 48 and "slope" in tl):
        return ("STEEP_SLOPE", True, True)
    if "slope" in tl or "wedge" in tl or "cheese" in tl:
        return ("PLANAR_SLOPE", True, True)
    if any(k in tl for k in ("round", "cylinder", "cone", "quarter", "corner")):
        return ("XY_WEDGE_ROUND", True, True)
    if dz == 8 and len(stud_z) == 0:
        return ("SMOOTH_TILE", True, True)
    if dz == 8:
        return ("CORE_PLATE", True, True)
    return ("CURVED_SLOPE" if fill_ratio < 0.85 else "CORE_BRICK", True, True)


@dataclass(frozen=True, slots=True)
class CatalogMoldEntry:
    """Canonical mechanical base mold entry in `CatalogShapeRetrievalIndex`."""

    mold_id: str
    title: str
    taxonomy_class: str
    license_tier: str  # "cc_by_4_0" | "cc_by_sa_4_0"
    is_structural_buildable: bool
    is_flush_system_mold: bool
    studs_x: int
    studs_y: int
    height_ldu: int
    num_anti_studs: int
    num_top_studs: int
    has_snot_ports: bool
    fill_ratio: float
    variant_count: int
    # Up to 4 C4 rotations: `(rot_id, sx_cells, sy_cells, sz_cells, flat_u64_tuple, normal_6d_tuple)`
    c4_footprints: Tuple[
        Tuple[int, int, int, int, Tuple[int, ...], Tuple[float, ...]],
        ...,
    ]


@dataclass(frozen=True, slots=True)
class CatalogRetrievalResult:
    """Result of online `CatalogShapeRetrievalIndex.retrieve_top_k(...)`."""

    retrieved_part_ids: Tuple[str, ...]
    sculptural_part_ids: Tuple[str, ...]
    backbone_part_ids: Tuple[str, ...]
    class_counts: Dict[str, int]
    motif_weights: Dict[str, float]
    license_filter: str
    query_wall_ms: float


def _build_c4_footprints(
    prim: BrickPrimitive,
) -> Tuple[Tuple[int, int, int, int, Tuple[int, ...], Tuple[float, ...]], ...]:
    """Precompute origin-aligned `uint64` cell bitmasks across the 4 planar `C4_Z_ROT_IDS` poses."""
    if not prim.hull_boxes:
        return ()
    anti_z = [
        p
        for p in prim.ports
        if p.polarity == PortPolarity.ANTI_STUD and p.local_frame.z_axis == (0, 0, 1)
    ]
    out: List[Tuple[int, int, int, int, Tuple[int, ...], Tuple[float, ...]]] = []
    seen_grids: Set[Tuple[int, int, int, Tuple[int, ...]]] = set()
    for rid in C4_Z_ROT_IDS:
        r_frame = LDUFrame(rid, (0, 0, 0))
        rboxes = [b.transform(r_frame) for b in prim.hull_boxes]
        min_bx = min(b.xmin for b in rboxes)
        min_by = min(b.ymin for b in rboxes)
        min_bz = min(b.zmin for b in rboxes)
        if anti_z:
            apts = [r_frame.compose(p.local_frame).pos for p in anti_z]
            ox = min(pt[0] for pt in apts) - 10
            oy = min(pt[1] for pt in apts) - 10
            oz = min(pt[2] for pt in apts)
            while ox > min_bx:
                ox -= 20
            while oy > min_by:
                oy -= 20
            while oz > min_bz:
                oz -= 24
        else:
            ox = min_bx
            oy = min_by
            oz = min_bz
        sboxes = [
            LDUBox(
                b.xmin - ox,
                b.ymin - oy,
                b.zmin - oz,
                b.xmax - ox,
                b.ymax - oy,
                b.zmax - oz,
            )
            for b in rboxes
        ]
        sx, sy, sz, u64_grid, n6 = rasterize_boxes_to_u64_cells(sboxes)
        flat_u64 = tuple(int(v) for v in u64_grid.ravel())
        sig = (sx, sy, sz, flat_u64)
        if sig in seen_grids:
            continue
        seen_grids.add(sig)
        out.append((int(rid), sx, sy, sz, flat_u64, tuple(round(float(x), 5) for x in n6)))
    return tuple(out)


class CatalogShapeRetrievalIndex:
    """
    Deduplicated canonical-mold `uint64` shape & semantic retrieval index over all `16,085`
    parts in `COMBINED_BRICK_CATALOG`.
    """

    def __init__(self, index_path: Path = DEFAULT_RETRIEVAL_INDEX_NPZ) -> None:
        self._index_path = Path(index_path)
        self._loaded = False
        self._index_sha256: str = "not_indexed"
        self._total_catalog_parts: int = 0
        self._entries: Tuple[CatalogMoldEntry, ...] = ()
        self._by_mold_id: Dict[str, CatalogMoldEntry] = {}
        self._buildable_entries: Tuple[CatalogMoldEntry, ...] = ()
        # Vectorized scoring arrays for `1x1x1` and multi-cell representative bitmasks
        self._cand_mold_ids: Tuple[str, ...] = ()
        self._cand_classes: Tuple[str, ...] = ()
        self._cand_is_cc_by_4: np.ndarray = np.zeros((0,), dtype=bool)
        self._cand_primary_u64: np.ndarray = np.zeros((0,), dtype=np.uint64)
        self._cand_normals_6d: np.ndarray = np.zeros((0, 6), dtype=np.float32)
        self._cand_fill_ratio: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._cand_height_ldu: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._cand_has_top_stud: np.ndarray = np.zeros((0,), dtype=bool)

    @property
    def is_available(self) -> bool:
        return self._index_path.is_file()

    @property
    def index_sha256(self) -> str:
        if not self._loaded and self._index_path.is_file():
            self._ensure_loaded()
        return self._index_sha256

    @property
    def total_catalog_parts(self) -> int:
        self._ensure_loaded()
        return self._total_catalog_parts

    @property
    def entries(self) -> Tuple[CatalogMoldEntry, ...]:
        self._ensure_loaded()
        return self._entries

    @property
    def num_molds(self) -> int:
        self._ensure_loaded()
        return len(self._entries)

    @property
    def buildable_entries(self) -> Tuple[CatalogMoldEntry, ...]:
        self._ensure_loaded()
        return self._buildable_entries

    def get_entry(self, mold_id: str) -> Optional[CatalogMoldEntry]:
        self._ensure_loaded()
        return self._by_mold_id.get(mold_id) or self._by_mold_id.get(mold_id.lower())

    def compile_and_save(self, output_path: Optional[Path] = None) -> Path:
        """
        Scan all `16,085` parts in `COMBINED_BRICK_CATALOG`, deduplicate mechanical molds,
        compute `uint64` C4 sub-voxel bitmasks & 6D surface normals, and write compact `.npz`.
        """
        target_path = Path(output_path) if output_path is not None else self._index_path
        COMPILED_LDRAW_CATALOG._ensure_loaded()
        all_pids = list(COMBINED_BRICK_CATALOG.keys())

        groups: Dict[Tuple[Any, ...], List[Tuple[str, str, BrickPrimitive, str]]] = {}
        for pid in all_pids:
            prim = get_brick_primitive(pid)
            if pid in EXTENDED_BRICK_CATALOG:
                title = prim.name
                lic = "cc_by_4_0"
            else:
                r_id = COMPILED_LDRAW_CATALOG.resolve_id(pid)
                rec = COMPILED_LDRAW_CATALOG._raw_parts.get(r_id or pid.lower(), {})
                title = str(rec.get("n", prim.name))
                tier_s = str(rec.get("t", ""))
                lic = "cc_by_sa_4_0" if tier_s.startswith("TIER_B") else "cc_by_4_0"

            # Keep curated EXTENDED sculptural/core parts distinct if needed, otherwise group exact mechanical twins
            sig = _canonical_mechanical_signature(prim)
            groups.setdefault(sig, []).append((pid, title, prim, lic))

        mold_entries: List[CatalogMoldEntry] = []
        for _, members in groups.items():
            members.sort(key=lambda item: _mold_id_preference_key(item[0], item[1]))
            best_pid, best_title, best_prim, best_lic = members[0]
            # If any unpatterned member has CC-BY-4.0 geometry/ports, prefer its license if best_pid is CC-BY-4.0
            tax_cls, buildable, is_flush = _classify_mold_and_hygiene(
                best_pid, best_title, best_prim, best_lic
            )
            if best_prim.hull_boxes:
                bx0 = min(b.xmin for b in best_prim.hull_boxes)
                bx1 = max(b.xmax for b in best_prim.hull_boxes)
                by0 = min(b.ymin for b in best_prim.hull_boxes)
                by1 = max(b.ymax for b in best_prim.hull_boxes)
                bz0 = min(b.zmin for b in best_prim.hull_boxes)
                bz1 = max(b.zmax for b in best_prim.hull_boxes)
                sx_s = max(1, int(round((bx1 - bx0) / 20.0)))
                sy_s = max(1, int(round((by1 - by0) / 20.0)))
                dz_l = max(1, bz1 - bz0)
                tot_v = sum(b.volume_ldu3 for b in best_prim.hull_boxes)
                fill_r = min(1.0, float(tot_v) / float(max(1, (bx1 - bx0) * (by1 - by0) * dz_l)))
            else:
                sx_s, sy_s, dz_l, fill_r = 1, 1, 8, 0.0

            anti_cnt = sum(1 for p in best_prim.ports if p.polarity == PortPolarity.ANTI_STUD)
            top_cnt = sum(
                1
                for p in best_prim.ports
                if p.polarity == PortPolarity.STUD and p.local_frame.z_axis == (0, 0, 1)
            )
            has_snot = any(p.local_frame.z_axis[2] == 0 for p in best_prim.ports)
            c4_fp = _build_c4_footprints(best_prim) if (buildable and is_flush) else ()

            mold_entries.append(
                CatalogMoldEntry(
                    mold_id=best_pid,
                    title=best_title,
                    taxonomy_class=tax_cls,
                    license_tier=best_lic,
                    is_structural_buildable=buildable,
                    is_flush_system_mold=is_flush,
                    studs_x=sx_s,
                    studs_y=sy_s,
                    height_ldu=dz_l,
                    num_anti_studs=anti_cnt,
                    num_top_studs=top_cnt,
                    has_snot_ports=has_snot,
                    fill_ratio=round(fill_r, 5),
                    variant_count=len(members),
                    c4_footprints=c4_fp,
                )
            )

        mold_entries.sort(key=lambda e: e.mold_id)
        # Deduplicate LDraw minor mold-revision letters (e.g. 3049a/3049b/3049c, 2412a/2412b)
        # when C4 uint64 sub-voxel footprints and port counts are identical, folding variant_count.
        rev_re = re.compile(r"^([0-9]{2,6})[a-d]$")
        dedup_map: Dict[Tuple[Any, ...], CatalogMoldEntry] = {}
        final_entries: List[CatalogMoldEntry] = []
        for e in mold_entries:
            m_rev = rev_re.match(e.mold_id)
            if m_rev and e.is_structural_buildable and e.c4_footprints and e.mold_id not in EXTENDED_BRICK_CATALOG:
                fp_sig = tuple((cx, cy, cz, u_t) for _, cx, cy, cz, u_t, _ in e.c4_footprints)
                rkey = (m_rev.group(1), e.taxonomy_class, e.num_anti_studs, e.num_top_studs, fp_sig)
                if rkey in dedup_map:
                    prev = dedup_map[rkey]
                    merged = CatalogMoldEntry(
                        mold_id=prev.mold_id,
                        title=prev.title,
                        taxonomy_class=prev.taxonomy_class,
                        license_tier="cc_by_4_0" if "cc_by_4_0" in (prev.license_tier, e.license_tier) else prev.license_tier,
                        is_structural_buildable=prev.is_structural_buildable,
                        is_flush_system_mold=prev.is_flush_system_mold,
                        studs_x=prev.studs_x,
                        studs_y=prev.studs_y,
                        height_ldu=prev.height_ldu,
                        num_anti_studs=prev.num_anti_studs,
                        num_top_studs=prev.num_top_studs,
                        has_snot_ports=prev.has_snot_ports,
                        fill_ratio=prev.fill_ratio,
                        variant_count=prev.variant_count + e.variant_count,
                        c4_footprints=prev.c4_footprints,
                    )
                    dedup_map[rkey] = merged
                    idx_prev = next(i for i, x in enumerate(final_entries) if x.mold_id == prev.mold_id)
                    final_entries[idx_prev] = merged
                    continue
                dedup_map[rkey] = e
            final_entries.append(e)
        mold_entries = final_entries

        meta_payload = {
            "schema_version": "1.0.0",
            "total_catalog_parts": len(all_pids),
            "num_canonical_molds": len(mold_entries),
            "molds": [
                {
                    "id": e.mold_id,
                    "n": e.title,
                    "c": e.taxonomy_class,
                    "l": e.license_tier,
                    "b": int(e.is_structural_buildable),
                    "f": int(e.is_flush_system_mold),
                    "sx": e.studs_x,
                    "sy": e.studs_y,
                    "dz": e.height_ldu,
                    "na": e.num_anti_studs,
                    "ns": e.num_top_studs,
                    "sn": int(e.has_snot_ports),
                    "fr": e.fill_ratio,
                    "vc": e.variant_count,
                    "fp": [
                        [r, cx, cy, cz, [str(u) for u in u_t], list(n_t)]
                        for r, cx, cy, cz, u_t, n_t in e.c4_footprints
                    ],
                }
                for e in mold_entries
            ],
        }
        json_bytes = json.dumps(meta_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

        # Build compact vectorized arrays over all buildable flush molds x C4 orientations
        v_mold_idx: List[int] = []
        v_u64: List[int] = []
        v_n6: List[List[float]] = []
        buildable_list = [e for e in mold_entries if e.is_structural_buildable and e.is_flush_system_mold]
        for b_idx, e in enumerate(buildable_list):
            for _, cx, cy, cz, u_t, n_t in e.c4_footprints:
                # Representative boundary cell is the least-filled non-zero cell (captures tapered/curved tip!)
                nz_cells = [u for u in u_t if u != 0]
                if not nz_cells:
                    continue
                nz_cells.sort(key=lambda u: bin(u).count("1"))
                v_mold_idx.append(b_idx)
                v_u64.append(nz_cells[0])
                v_n6.append(list(n_t))

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            meta_json=np.frombuffer(json_bytes, dtype=np.uint8),
            pose_mold_idx=np.asarray(v_mold_idx, dtype=np.int32),
            pose_rep_u64=np.asarray(v_u64, dtype=np.uint64),
            pose_normals_6d=np.asarray(v_n6, dtype=np.float32),
        )
        raw_npz = buf.getvalue()
        target_path.write_bytes(raw_npz)
        self._loaded = False
        self._index_path = target_path
        self._ensure_loaded()
        return target_path

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._index_path.is_file():
            self.compile_and_save(self._index_path)
            return
        raw = self._index_path.read_bytes()
        self._index_sha256 = hashlib.sha256(raw).hexdigest()
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
            pose_mold_idx = np.asarray(data["pose_mold_idx"], dtype=np.int32)
            pose_rep_u64 = np.asarray(data["pose_rep_u64"], dtype=np.uint64)
            pose_normals_6d = np.asarray(data["pose_normals_6d"], dtype=np.float32)

        self._total_catalog_parts = int(meta["total_catalog_parts"])
        entries: List[CatalogMoldEntry] = []
        for m in meta["molds"]:
            fps = tuple(
                (
                    int(fp[0]),
                    int(fp[1]),
                    int(fp[2]),
                    int(fp[3]),
                    tuple(int(u_s) for u_s in fp[4]),
                    tuple(float(nf) for nf in fp[5]),
                )
                for fp in m["fp"]
            )
            entries.append(
                CatalogMoldEntry(
                    mold_id=str(m["id"]),
                    title=str(m["n"]),
                    taxonomy_class=str(m["c"]),
                    license_tier=str(m["l"]),
                    is_structural_buildable=bool(m["b"]),
                    is_flush_system_mold=bool(m["f"]),
                    studs_x=int(m["sx"]),
                    studs_y=int(m["sy"]),
                    height_ldu=int(m["dz"]),
                    num_anti_studs=int(m["na"]),
                    num_top_studs=int(m["ns"]),
                    has_snot_ports=bool(m["sn"]),
                    fill_ratio=float(m["fr"]),
                    variant_count=int(m["vc"]),
                    c4_footprints=fps,
                )
            )
        self._entries = tuple(entries)
        self._by_mold_id = {e.mold_id: e for e in self._entries}
        self._buildable_entries = tuple(
            e for e in self._entries if e.is_structural_buildable and e.is_flush_system_mold
        )

        # Cache per-pose vectorized arrays
        self._pose_mold_idx = pose_mold_idx
        self._pose_rep_u64 = pose_rep_u64
        self._pose_normals_6d = pose_normals_6d
        self._buildable_is_cc4 = np.array(
            [e.license_tier == "cc_by_4_0" for e in self._buildable_entries], dtype=bool
        )
        self._buildable_fill = np.array(
            [e.fill_ratio for e in self._buildable_entries], dtype=np.float32
        )
        self._buildable_dz = np.array(
            [e.height_ldu for e in self._buildable_entries], dtype=np.int32
        )
        valid_sculpt: List[bool] = []
        for e in self._buildable_entries:
            prim_e = get_brick_primitive(e.mold_id)
            nb = len(prim_e.hull_boxes)
            t_low = e.title.lower()
            is_non_slope_decor = e.taxonomy_class in ("CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE") and any(
                kw in t_low for kw in ("container", "crate", "fence", "picket", "window", "door", "bar ")
            )
            all_coarse_20x24 = nb >= 1 and all(
                (b.xmax - b.xmin) % 20 == 0
                and (b.ymax - b.ymin) % 20 == 0
                and (b.zmax - b.zmin) % 24 == 0
                for b in prim_e.hull_boxes
            )
            if (
                is_non_slope_decor
                or (
                    nb == 1
                    and (
                        e.taxonomy_class
                        in ("ARCH", "INVERTED_SLOPE", "CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE", "THIN_PANEL")
                        or (e.taxonomy_class == "XY_WEDGE_ROUND" and e.studs_x * e.studs_y > 1)
                    )
                )
                or (
                    e.taxonomy_class
                    in ("ARCH", "INVERTED_SLOPE", "CURVED_SLOPE", "PLANAR_SLOPE", "STEEP_SLOPE", "THIN_PANEL", "XY_WEDGE_ROUND")
                    and (
                        (e.fill_ratio > 0.90 and e.studs_x * e.studs_y > 1)
                        or (all_coarse_20x24 and e.taxonomy_class in ("STEEP_SLOPE", "PLANAR_SLOPE", "CURVED_SLOPE"))
                    )
                )
            ):
                valid_sculpt.append(False)
            else:
                valid_sculpt.append(True)
        self._valid_sculpt_mask = np.array(valid_sculpt, dtype=bool)
        self._loaded = True

    def retrieve_top_k(
        self,
        occ_5x5x6: np.ndarray,
        top_k: int = 32,
        license_filter: str = "all",
        semantic_style_hint: Optional[str] = None,
    ) -> CatalogRetrievalResult:
        """
        Retrieve a mesh-conditioned `top-K` (`default K = 32`) open-catalog kit in `< 3 ms`
        using vectorized `uint64` boundary-cell bitmask matching + 6D surface-normal cosine scoring.
        """
        self._ensure_loaded()
        t0 = time.perf_counter()
        if license_filter not in ("all", "cc_by_4_0_only"):
            raise ValueError(f"Unsupported license_filter={license_filter!r}; expected 'all' or 'cc_by_4_0_only'.")

        eff_k = max(12, int(top_k))
        t_u64 = pack_occ_5x5x6_to_u64(occ_5x5x6)
        pop_cells = popcount_u64_array(t_u64)

        # Identify boundary cells (`0 < pop < 64` or adjacent to empty space)
        boundary_mask = (pop_cells > 0) & (pop_cells < 64)
        if not np.any(boundary_mask):
            pad_f = np.pad(pop_cells == 64, 1, mode="constant", constant_values=False)
            c_f = pad_f[1:-1, 1:-1, 1:-1]
            ero = (
                c_f
                & pad_f[2:, 1:-1, 1:-1]
                & pad_f[:-2, 1:-1, 1:-1]
                & pad_f[1:-1, 2:, 1:-1]
                & pad_f[1:-1, :-2, 1:-1]
                & pad_f[1:-1, 1:-1, 2:]
                & pad_f[1:-1, 1:-1, :-2]
            )
            boundary_mask = (pop_cells > 0) & (~ero)

        b_u64 = t_u64[boundary_mask]
        if b_u64.size > 48:
            # Deterministic stride sampling of up to 48 representative partial boundary cells for `< 2 ms` latency
            step = max(1, b_u64.size // 48)
            b_u64 = b_u64[::step][:48]

        # Extract mesh-level geometric motif weights directly from sub-voxel bitmasks
        low_z_mask = np.uint64(0x00000000FFFFFFFF)
        high_z_mask = np.uint64(0xFFFFFFFF00000000)
        q00_xy = np.uint64(0x0033003300330033)
        q11_xy = np.uint64(0xCC00CC00CC00CC00)

        if b_u64.size > 0:
            low_pop = popcount_u64_array(b_u64 & low_z_mask)
            high_pop = popcount_u64_array(b_u64 & high_z_mask)
            q00_pop = popcount_u64_array(b_u64 & q00_xy)
            q11_pop = popcount_u64_array(b_u64 & q11_xy)
            tot_b = float(max(1, b_u64.size))
            w_top_taper = float(np.count_nonzero(low_pop > high_pop + 4)) / tot_b
            w_inv_overhang = float(np.count_nonzero(high_pop > low_pop + 4)) / tot_b
            w_xy_diag = float(np.count_nonzero(np.abs(q00_pop - q11_pop) >= 6)) / tot_b
            w_thin = float(np.count_nonzero((low_pop + high_pop) <= 24)) / tot_b
        else:
            w_top_taper, w_inv_overhang, w_xy_diag, w_thin = 0.25, 0.05, 0.15, 0.10

        # Detect half-stud (`10 LDU`, i.e., `sub_idx % 4 == 2`) silhouette transitions across courses
        occ_b = np.asarray(occ_5x5x6, dtype=bool)
        half_stud_hits = 0
        active_slices = 0
        for iz in range(0, occ_b.shape[2], 2):
            sl = occ_b[:, :, iz]
            if np.any(sl):
                active_slices += 1
                xs, ys = np.nonzero(sl)
                if (int(xs.min()) % 4 == 2) or (int(xs.max() + 1) % 4 == 2) or (int(ys.min()) % 4 == 2) or (int(ys.max() + 1) % 4 == 2):
                    half_stud_hits += 1
        w_jumper = float(half_stud_hits) / float(max(1, active_slices))

        motif_weights = {
            "top_taper": round(w_top_taper, 4),
            "underside_overhang": round(w_inv_overhang, 4),
            "xy_diagonal_round": round(w_xy_diag, 4),
            "thin_strut_wall": round(w_thin, 4),
            "half_stud_phase": round(w_jumper, 4),
        }

        # Vectorized `uint64` bitwise IoU gain scoring across all buildable canonical molds!
        n_build = len(self._buildable_entries)
        mold_scores = np.full((n_build,), -1e6, dtype=np.float32)

        if b_u64.size > 0 and self._pose_rep_u64.size > 0:
            # `(P, B)` bitwise AND & overshoot popcount
            inter = popcount_u64_array(self._pose_rep_u64[:, None] & b_u64[None, :])
            over = popcount_u64_array(self._pose_rep_u64[:, None] & (~b_u64[None, :]))
            net_gain = (2.0 * inter.astype(np.float32) - 1.35 * over.astype(np.float32)) / 64.0
            # Top-4 boundary cell match average per pose
            k_top = min(4, net_gain.shape[1])
            part_top = np.partition(net_gain, net_gain.shape[1] - k_top, axis=1)[:, -k_top:]
            pose_sc = np.mean(part_top, axis=1)
            np.maximum.at(mold_scores, self._pose_mold_idx, pose_sc)
        else:
            mold_scores[:] = 0.0

        # Apply license filter & motif/semantic class priors
        cc4_only = license_filter == "cc_by_4_0_only"

        for idx, e in enumerate(self._buildable_entries):
            if not self._valid_sculpt_mask[idx]:
                mold_scores[idx] = -1e9
                continue
            if cc4_only and not self._buildable_is_cc4[idx]:
                mold_scores[idx] = -1e9
                continue
            cls = e.taxonomy_class
            n_boxes_e = len(get_brick_primitive(e.mold_id).hull_boxes)
            bonus = 0.0
            if cls == "CURVED_SLOPE":
                bonus += 0.36 * w_top_taper + 0.08
                if n_boxes_e >= 2:
                    bonus += 0.18 + (0.06 if e.height_ldu in (8, 16) else 0.0)
            elif cls == "PLANAR_SLOPE":
                bonus += 0.36 * w_top_taper
                if n_boxes_e >= 2:
                    bonus += 0.18 + (0.04 if e.height_ldu in (8, 16) else 0.0)
            elif cls == "STEEP_SLOPE":
                bonus += 0.30 * (w_top_taper + 0.5 * w_thin)
                if n_boxes_e >= 2:
                    bonus += 0.16
            elif cls == "INVERTED_SLOPE":
                bonus += 0.45 * w_inv_overhang
                if e.num_top_studs == 0:
                    bonus -= 0.35
            elif cls == "ARCH":
                bonus += 0.42 * w_inv_overhang
            elif cls == "XY_WEDGE_ROUND":
                bonus += 0.40 * w_xy_diag
                if n_boxes_e >= 2 and e.studs_x * e.studs_y <= 9:
                    bonus += 0.16
            elif cls == "THIN_PANEL":
                bonus += 0.44 * w_thin
                if n_boxes_e >= 2 and e.fill_ratio <= 0.55 and e.studs_x * e.studs_y <= 4:
                    bonus += 0.18
            elif cls == "JUMPER_HALF_STUD":
                bonus += 0.58 * max(w_jumper, 0.5 * w_thin)
            elif cls == "SNOT_ANCHOR":
                bonus += 0.34 * max(w_thin, w_xy_diag, w_top_taper)
            elif cls == "SMOOTH_TILE":
                bonus += 0.18
            # Prefer compact, versatile footprints (1x1..2x6) and strongly penalize oversized niche baseplates/cockpits
            area_s = e.studs_x * e.studs_y
            if area_s > 16 or e.studs_x > 8 or e.studs_y > 8:
                bonus -= 0.85
            elif area_s == 1:
                bonus += 0.16
            elif area_s == 2:
                bonus += 0.19
            elif area_s <= 4:
                bonus += 0.16
            elif area_s <= 8:
                bonus += 0.10
            elif area_s <= 12:
                bonus += 0.02
            else:
                bonus -= 0.25
            mold_scores[idx] += bonus

        # 1. Build a strictly nested prefix ranking (`Top-16 ⊂ Top-32 ⊂ Top-64`) for monotonic K-ablation:
        #    Positions 0..9: minimal 10-part brick+plate structural backbone (`1x1..2x6` bricks & `1x1..2x4` plates)
        #    so `Top-16` already includes `6` high-leverage sculptural/jumper/slope/SNOT molds!
        essential_backbone = (
            "3005", "3004", "3001", "2456",
            "3024", "3023", "3710", "3020", "3795", "3666",
        )
        extended_backbone = (
            "3010", "3003", "3002", "3021", "3022", "3623",
            "3622", "3009", "3008", "3460", "3007", "3034",
        )

        cc4_only = license_filter == "cc_by_4_0_only"
        ranked_indices = np.argsort(-mold_scores)
        by_cls_ranked: Dict[str, List[str]] = {c: [] for c in MOLD_TAXONOMY_CLASSES}
        score_by_mid: Dict[str, float] = {}
        for idx_r in ranked_indices:
            sc_r = float(mold_scores[int(idx_r)])
            if sc_r <= -1e8:
                continue
            ent_r = self._buildable_entries[int(idx_r)]
            by_cls_ranked[ent_r.taxonomy_class].append(ent_r.mold_id)
            score_by_mid[ent_r.mold_id] = sc_r

        # Ensure verified `5x5x6`-sliced multi-box slopes, curves, facets, panels, SNOT bricks & tiles lead each class,
        # dynamically ranked by this mesh's 40-D geometry descriptor (`score_by_mid`) after the primary class anchors.
        head_anchor_counts = {
            "CURVED_SLOPE": 8,
            "PLANAR_SLOPE": 8,
            "STEEP_SLOPE": 3,
            "XY_WEDGE_ROUND": 5,
            "SMOOTH_TILE": 7,
            "SNOT_ANCHOR": 7,
            "THIN_PANEL": 4,
        }
        for cls_k, workhorses in (
            ("JUMPER_HALF_STUD", ("15573",)),
            (
                "CURVED_SLOPE",
                (
                    "44675", "49656", "79757", "49307", "11477", "15068", "88930", "6091",
                    "47457", "41855", "50950", "93604", "88293",
                ),
            ),
            (
                "PLANAR_SLOPE",
                (
                    "54200", "85984", "3040a", "13548", "3043", "3039", "962", "3044a", "3038", "3298",
                    "92946", "4161", "6565", "3675", "99301", "23949", "3297", "3041", "3042",
                ),
            ),
            ("STEEP_SLOPE", ("3678b", "78886", "3684c")),
            ("INVERTED_SLOPE", ("3665", "3660", "4287c", "3049b")),
            ("ARCH", ("3659", "15254", "3308", "6182", "6005")),
            ("THIN_PANEL", ("4864a", "2362a", "4865a", "23950", "30413", "4215a", "6231", "93095")),
            (
                "XY_WEDGE_ROUND",
                (
                    "25269", "24307", "3063a", "27925", "24299", "27928", "2450",
                    "2462", "2420", "87620", "1126", "26601", "2357",
                ),
            ),
            (
                "SMOOTH_TILE",
                (
                    "3070b", "3069b", "2431", "3068b", "87079", "63864",
                    "26603", "6636", "69729", "4162",
                ),
            ),
            ("SNOT_ANCHOR", ("30414", "87087", "99780", "44728", "4070", "36841", "11211", "99207", "99206")),
        ):
            cur_l = by_cls_ranked[cls_k]
            wh_index = {w: idx_w for idx_w, w in enumerate(workhorses)}
            promoted = [
                w
                for w in workhorses
                if w in self._by_mold_id
                and (w in score_by_mid or w == "36841")
                and (not cc4_only or self._by_mold_id[w].license_tier == "cc_by_4_0")
            ]
            n_head = head_anchor_counts.get(cls_k, 2)
            if len(promoted) > n_head:
                head_w = promoted[:n_head]
                tail_w = sorted(
                    promoted[n_head:],
                    key=lambda w: (-score_by_mid.get(w, 0.0), wh_index[w]),
                )
                promoted = head_w + tail_w
            compact_cur = [
                m_id
                for m_id in cur_l
                if (
                    self._by_mold_id[m_id].studs_x <= 8
                    and self._by_mold_id[m_id].studs_y <= 8
                    and self._by_mold_id[m_id].studs_x * self._by_mold_id[m_id].studs_y <= 16
                )
            ]
            large_cur = [m_id for m_id in cur_l if m_id not in set(compact_cur)]
            merged_c: List[str] = []
            seen_m: Set[str] = set()
            for seq_m in (promoted, compact_cur, large_cur):
                for m_id in seq_m:
                    if m_id not in seen_m:
                        merged_c.append(m_id)
                        seen_m.add(m_id)
            by_cls_ranked[cls_k] = merged_c

        ordered_full: List[str] = []
        chosen_set: Set[str] = set()
        backbone_list: List[str] = []

        for bpid in essential_backbone:
            ent = self._by_mold_id.get(bpid)
            if ent is not None and (not cc4_only or ent.license_tier == "cc_by_4_0"):
                ordered_full.append(bpid)
                chosen_set.add(bpid)
                backbone_list.append(bpid)

        # Order sculptural classes dynamically by this mesh's boundary motif weights (independent of `top_k`
        # so `Top-16 ⊂ Top-32 ⊂ Top-48 ⊂ Top-64` is an exact prefix identity!)
        occ_sub_cnt = int(np.count_nonzero(occ_b))
        is_sparse_thin = (w_thin >= 0.52) or (occ_sub_cnt < 1000)
        needs_snot_skin = (w_thin >= 0.06) or (w_jumper >= 0.03) or (w_top_taper >= 0.08)
        cls_priority: List[Tuple[float, str]] = [
            (0.92 if (w_jumper > 0.02 or is_sparse_thin) else 0.76, "JUMPER_HALF_STUD"),
            (0.88 + 0.5 * w_top_taper, "CURVED_SLOPE"),
            (0.87 + 0.5 * w_top_taper, "PLANAR_SLOPE"),
            (0.80 + 0.6 * w_xy_diag, "XY_WEDGE_ROUND"),
            (0.875 if occ_sub_cnt < 900 else 0.74, "SMOOTH_TILE"),
            (0.90 if w_inv_overhang >= 0.03 else (0.38 + 1.35 * w_inv_overhang), "INVERTED_SLOPE"),
            (0.89 if w_inv_overhang >= 0.03 else (0.35 + 1.25 * w_inv_overhang), "ARCH"),
            (0.68 + 0.55 * (w_top_taper + w_thin), "STEEP_SLOPE"),
            (0.885 if is_sparse_thin else (0.85 if needs_snot_skin else (0.50 + 0.55 * w_xy_diag)), "SNOT_ANCHOR"),
            (0.96 if is_sparse_thin else (0.36 + 0.7 * w_thin), "THIN_PANEL"),
        ]
        cls_priority.sort(key=lambda x: (-x[0], x[1]))
        # Place `EXT_BACKBONE` (`3010, 3003, 3002, 3622, 3021, 3022, 3623`) after Round 0's top-6 sculptural heads
        # by inserting it at index 6 so `Top-16` still holds 6 sculptural classes while `Top-48` ALWAYS retains
        # all 7 medium plates/bricks even on sparse thin-panel + arch meshes!
        ext_bb_stream = [
            b
            for b in extended_backbone
            if b in self._by_mold_id and (not cc4_only or self._by_mold_id[b].license_tier == "cc_by_4_0")
        ]
        stream_names: List[str] = [c_n for _, c_n in cls_priority]
        round_streams: List[List[str]] = [list(by_cls_ranked[c_n]) for c_n in stream_names]
        ins_pos_bb = min(6, len(stream_names))
        stream_names.insert(ins_pos_bb, "EXT_BACKBONE")
        round_streams.insert(ins_pos_bb, ext_bb_stream)

        stream_caps: Dict[str, int] = {
            "JUMPER_HALF_STUD": 1,
            "ARCH": 2 if w_inv_overhang >= 0.03 else 1,
            "INVERTED_SLOPE": 3 if w_inv_overhang >= 0.03 else 1,
            "STEEP_SLOPE": 3,
            "SNOT_ANCHOR": 7,
            "THIN_PANEL": 5 if is_sparse_thin else 1,
            "SMOOTH_TILE": 8,
            "EXT_BACKBONE": 7,
        }
        high_volume_streams = {
            "PLANAR_SLOPE",
            "CURVED_SLOPE",
            "SMOOTH_TILE",
            "EXT_BACKBONE",
        }
        if is_sparse_thin:
            high_volume_streams.add("THIN_PANEL")
        target_build_len = max(eff_k, 64)

        cursors = [0] * len(round_streams)
        picked_per_stream = [0] * len(round_streams)
        round_idx = 0
        while len(ordered_full) < target_build_len:
            progressed = False
            for s_i, stream in enumerate(round_streams):
                sname = stream_names[s_i]
                cap_s = stream_caps.get(sname, 10_000)
                if picked_per_stream[s_i] >= cap_s:
                    continue
                if sname == "SMOOTH_TILE" and round_idx == 1:
                    n_draw = 4 if occ_sub_cnt < 900 else 6
                elif sname == "SMOOTH_TILE" and round_idx == 2:
                    n_draw = 1
                elif sname == "SMOOTH_TILE" and round_idx == 3:
                    n_draw = 1
                elif sname == "SNOT_ANCHOR" and round_idx == 1:
                    n_draw = 5
                elif sname == "SNOT_ANCHOR" and round_idx == 2:
                    n_draw = 1
                elif sname == "SNOT_ANCHOR" and round_idx == 3:
                    n_draw = 1
                elif sname == "EXT_BACKBONE" and round_idx == 1:
                    n_draw = 5
                elif sname == "EXT_BACKBONE" and round_idx == 2:
                    n_draw = 0
                elif sname == "EXT_BACKBONE" and round_idx == 3:
                    n_draw = 2
                elif sname == "CURVED_SLOPE" and round_idx == 1:
                    n_draw = 4
                elif sname == "CURVED_SLOPE" and round_idx == 2:
                    n_draw = 2
                elif sname == "PLANAR_SLOPE" and round_idx == 1:
                    n_draw = 4
                elif sname == "PLANAR_SLOPE" and round_idx == 2:
                    n_draw = 2
                elif sname == "XY_WEDGE_ROUND" and round_idx == 1:
                    n_draw = 1
                elif sname == "STEEP_SLOPE" and round_idx == 1:
                    n_draw = 2
                elif round_idx >= 1 and sname in high_volume_streams:
                    n_draw = 2
                else:
                    n_draw = 1
                for _ in range(n_draw):
                    if picked_per_stream[s_i] >= cap_s or len(ordered_full) >= target_build_len:
                        break
                    while cursors[s_i] < len(stream) and stream[cursors[s_i]] in chosen_set:
                        cursors[s_i] += 1
                    if cursors[s_i] < len(stream):
                        mid = stream[cursors[s_i]]
                        cursors[s_i] += 1
                        picked_per_stream[s_i] += 1
                        ordered_full.append(mid)
                        chosen_set.add(mid)
                        if mid in extended_backbone:
                            backbone_list.append(mid)
                        progressed = True
            round_idx += 1
            if not progressed:
                break

        for idx_r in ranked_indices:
            if len(ordered_full) >= target_build_len:
                break
            if mold_scores[int(idx_r)] <= -1e8:
                break
            mid = self._buildable_entries[int(idx_r)].mold_id
            if mid not in chosen_set:
                ordered_full.append(mid)
                chosen_set.add(mid)

        chosen = ordered_full[:eff_k]
        backbone_ids = tuple(m for m in chosen if m in backbone_list)

        cls_counts: Dict[str, int] = {}
        sculptural_ids: List[str] = []
        for mid in chosen:
            ent_c = self._by_mold_id.get(mid)
            if ent_c is not None:
                cls_counts[ent_c.taxonomy_class] = cls_counts.get(ent_c.taxonomy_class, 0) + 1
                if ent_c.taxonomy_class in SCULPTURAL_TAXONOMY_CLASSES:
                    sculptural_ids.append(mid)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return CatalogRetrievalResult(
            retrieved_part_ids=tuple(chosen),
            sculptural_part_ids=tuple(sculptural_ids),
            backbone_part_ids=backbone_ids,
            class_counts=cls_counts,
            motif_weights=motif_weights,
            license_filter=license_filter,
            query_wall_ms=dt_ms,
        )


CATALOG_RETRIEVAL_INDEX = CatalogShapeRetrievalIndex()
