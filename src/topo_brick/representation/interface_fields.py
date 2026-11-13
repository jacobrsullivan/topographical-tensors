"""
WP-L2: Topographical Frontier & Candidate Field Generator.

Projects exact geometric feasibility, persistent contact-graph seam topology,
static cantilever/shear proxies, local gravity, and target-shape occupancy onto:
1. Active Connection Frontier ports: `port_features` in R^{|F_t| x 24}
2. Sparse relative-frame frontier edges: `edge_index`, `edge_rel_rot_id`, `edge_rel_pos`, `edge_attr`
3. Verified course candidates: `candidate_features` in R^{M x 18}

All features are derived from the persistent `AssemblyState` (`PlacedPartTable`, `ContactGraph`,
`SpatialCollisionIndex`) and the exact `CandidatePool` (`X = T_u Q_θ F_p^-1`), avoiding heuristic
axis-only clearance ray blindspots.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.interface.brick_catalog import (
    BRICK_CATALOG,
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.ldu_frames import (
    BRICK_HEIGHT_LDU,
    C4_Z_ROT_IDS,
    LDUBox,
    STUD_PITCH_LDU,
    relative_transform,
)
from topo_brick.interface.port_manifold import (
    ActiveConnectionFrontier,
    AssemblyState,
    CandidatePool,
)

FRONTIER_PORT_CHANNELS: int = 24
CANDIDATE_FEATURE_CHANNELS: int = 18


class IntegralTargetGrid:
    """
    WP-L8b: O(1) 3D Summed-Area Table (Integral Volume Grid) and Spatial Cell Index
    over the integer LDU lattice.
    Precomputes a 3D prefix-sum array at 2-LDU resolution (the GCD of all catalog hull
    and SNOT bracket coordinates) so every candidate box intersection query executes in
    O(1) time (8 array lookups) with 100% bit-exact integer equivalence to pairwise
    LDUBox intersection.
    """

    def __init__(self, boxes: Sequence[LDUBox], step_ldu: int = 2) -> None:
        self.boxes: Tuple[LDUBox, ...] = tuple(boxes)
        self.step: int = step_ldu
        self.total_volume_ldu3: int = sum(b.volume_ldu3 for b in self.boxes)

        if not self.boxes:
            self.xmin = self.ymin = self.zmin = 0
            self.xmax = self.ymax = self.zmax = 0
            self.can_use_grid = False
            self.prefix_vol = np.zeros((1, 1, 1), dtype=np.int64)
            self._cell_map: Dict[Tuple[int, int, int], List[LDUBox]] = {}
            return

        self.xmin = min(b.xmin for b in self.boxes)
        self.ymin = min(b.ymin for b in self.boxes)
        self.zmin = min(b.zmin for b in self.boxes)
        self.xmax = max(b.xmax for b in self.boxes)
        self.ymax = max(b.ymax for b in self.boxes)
        self.zmax = max(b.zmax for b in self.boxes)

        # Build O(1) spatial hash cell map (20 x 20 x 24 LDU cells) for contains_point(pt)
        self._cell_map = {}
        for b in self.boxes:
            cx0, cx1 = b.xmin // 20, b.xmax // 20
            cy0, cy1 = b.ymin // 20, b.ymax // 20
            cz0, cz1 = b.zmin // 24, b.zmax // 24
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    for cz in range(cz0, cz1 + 1):
                        self._cell_map.setdefault((cx, cy, cz), []).append(b)

        all_divisible = all(
            (b.xmin % step_ldu == 0)
            and (b.xmax % step_ldu == 0)
            and (b.ymin % step_ldu == 0)
            and (b.ymax % step_ldu == 0)
            and (b.zmin % step_ldu == 0)
            and (b.zmax % step_ldu == 0)
            for b in self.boxes
        )
        nx = (self.xmax - self.xmin) // step_ldu
        ny = (self.ymax - self.ymin) // step_ldu
        nz = (self.zmax - self.zmin) // step_ldu

        if all_divisible and nx > 0 and ny > 0 and nz > 0 and (nx * ny * nz <= 8_000_000):
            self.can_use_grid = True
            voxel_vol = int(step_ldu * step_ldu * step_ldu)
            occ = np.zeros((nx, ny, nz), dtype=np.int64)
            for b in self.boxes:
                ix0 = (b.xmin - self.xmin) // step_ldu
                ix1 = (b.xmax - self.xmin) // step_ldu
                iy0 = (b.ymin - self.ymin) // step_ldu
                iy1 = (b.ymax - self.ymin) // step_ldu
                iz0 = (b.zmin - self.zmin) // step_ldu
                iz1 = (b.zmax - self.zmin) // step_ldu
                occ[ix0:ix1, iy0:iy1, iz0:iz1] += voxel_vol

            self.prefix_vol = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.int64)
            self.prefix_vol[1:, 1:, 1:] = occ.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
        else:
            self.can_use_grid = False
            self.prefix_vol = np.zeros((1, 1, 1), dtype=np.int64)

    def _pairwise_box_volume(self, qb: LDUBox) -> int:
        vol = 0
        for tb in self.boxes:
            ix = min(qb.xmax, tb.xmax) - max(qb.xmin, tb.xmin)
            if ix <= 0:
                continue
            iy = min(qb.ymax, tb.ymax) - max(qb.ymin, tb.ymin)
            if iy <= 0:
                continue
            iz = min(qb.zmax, tb.zmax) - max(qb.zmin, tb.zmin)
            if iz <= 0:
                continue
            vol += ix * iy * iz
        return vol

    def query_box_volume(self, qb: LDUBox) -> int:
        """O(1) 8-corner prefix-sum lookup for a single LDUBox `qb`."""
        if not self.boxes:
            return 0
        x0 = max(qb.xmin, self.xmin)
        x1 = min(qb.xmax, self.xmax)
        if x1 <= x0:
            return 0
        y0 = max(qb.ymin, self.ymin)
        y1 = min(qb.ymax, self.ymax)
        if y1 <= y0:
            return 0
        z0 = max(qb.zmin, self.zmin)
        z1 = min(qb.zmax, self.zmax)
        if z1 <= z0:
            return 0

        s = self.step
        if (
            self.can_use_grid
            and (x0 - self.xmin) % s == 0
            and (x1 - self.xmin) % s == 0
            and (y0 - self.ymin) % s == 0
            and (y1 - self.ymin) % s == 0
            and (z0 - self.zmin) % s == 0
            and (z1 - self.zmin) % s == 0
        ):
            ix0 = (x0 - self.xmin) // s
            ix1 = (x1 - self.xmin) // s
            iy0 = (y0 - self.ymin) // s
            iy1 = (y1 - self.ymin) // s
            iz0 = (z0 - self.zmin) // s
            iz1 = (z1 - self.zmin) // s
            p = self.prefix_vol
            return int(
                p[ix1, iy1, iz1]
                - p[ix0, iy1, iz1]
                - p[ix1, iy0, iz1]
                - p[ix1, iy1, iz0]
                + p[ix0, iy0, iz1]
                + p[ix0, iy1, iz0]
                + p[ix1, iy0, iz0]
                - p[ix0, iy0, iz0]
            )
        return self._pairwise_box_volume(qb)

    def intersection_volume(self, query_boxes: Sequence[LDUBox]) -> int:
        """Compute exact integer LDU^3 volume of `query_boxes` intersecting this target grid."""
        return sum(self.query_box_volume(qb) for qb in query_boxes)

    def contains_point(self, pt: Tuple[int, int, int]) -> bool:
        """O(1) strict interior point containment query via spatial hash cell lookup."""
        x, y, z = pt
        if x <= self.xmin or x >= self.xmax or y <= self.ymin or y >= self.ymax or z <= self.zmin or z >= self.zmax:
            return False
        cell_boxes = self._cell_map.get((x // 20, y // 20, z // 24))
        if not cell_boxes:
            return False
        return any(b.contains_point_strict(pt) for b in cell_boxes)


class TargetShapeBoxes:
    """
    Target 3D shape specification represented as a collection of non-overlapping integer LDU boxes.
    Backed automatically by `IntegralTargetGrid` (WP-L8b) for O(1) intersection volume and point queries
    with 100% bit-exact equivalence to pairwise box scanning.
    """

    __slots__ = ("boxes", "_grid")

    def __init__(self, boxes: Sequence[LDUBox]) -> None:
        object.__setattr__(self, "boxes", tuple(boxes))
        object.__setattr__(self, "_grid", IntegralTargetGrid(self.boxes))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("TargetShapeBoxes is immutable.")

    def __copy__(self) -> TargetShapeBoxes:
        return self

    def __deepcopy__(self, memo: object) -> TargetShapeBoxes:
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TargetShapeBoxes):
            return NotImplemented
        return self.boxes == other.boxes

    def __hash__(self) -> int:
        return hash(self.boxes)

    def __repr__(self) -> str:
        return f"TargetShapeBoxes(boxes={self.boxes!r})"

    @property
    def integral_grid(self) -> IntegralTargetGrid:
        return self._grid

    @property
    def total_volume_ldu3(self) -> int:
        return self._grid.total_volume_ldu3

    def intersection_volume(self, query_boxes: Sequence[LDUBox]) -> int:
        """Compute exact integer LDU^3 volume of `query_boxes` intersecting this target shape in O(1) per box."""
        return self._grid.intersection_volume(query_boxes)

    def intersection_volume_pairwise_reference(self, query_boxes: Sequence[LDUBox]) -> int:
        """Reference O(|query_boxes| * |target_boxes|) pairwise intersection loop for equivalence verification."""
        return sum(self._grid._pairwise_box_volume(qb) for qb in query_boxes)

    def contains_point(self, pt: Tuple[int, int, int]) -> bool:
        return self._grid.contains_point(pt)

    def iou_with_placed_boxes(self, placed_boxes: Sequence[LDUBox]) -> float:
        if not placed_boxes and self.total_volume_ldu3 == 0:
            return 1.0
        placed_vol = sum(b.volume_ldu3 for b in placed_boxes)
        inter_vol = self.intersection_volume(placed_boxes)
        union_vol = self.total_volume_ldu3 + placed_vol - inter_vol
        return float(inter_vol) / float(max(union_vol, 1))

    def precision_of_placed_boxes(self, placed_boxes: Sequence[LDUBox]) -> float:
        if not placed_boxes:
            return 1.0
        placed_vol = sum(b.volume_ldu3 for b in placed_boxes)
        inter_vol = self.intersection_volume(placed_boxes)
        return float(inter_vol) / float(max(placed_vol, 1))

    def occupied_bounds_ldu(self) -> Tuple[int, int, int, int, int, int]:
        if not self.boxes:
            return (0, 0, 0, 0, 0, 0)
        return (
            min(b.xmin for b in self.boxes),
            min(b.ymin for b in self.boxes),
            min(b.zmin for b in self.boxes),
            max(b.xmax for b in self.boxes),
            max(b.ymax for b in self.boxes),
            max(b.zmax for b in self.boxes),
        )

    def band_volume_ldu3(self, z0: int, z1: int) -> int:
        if not self.boxes or z1 <= z0:
            return 0
        xmin, ymin, _, xmax, ymax, _ = self.occupied_bounds_ldu()
        return self.intersection_volume((LDUBox(xmin, ymin, z0, xmax, ymax, z1),))

    def slice_z_band(self, z0: int, z1: int) -> TargetShapeBoxes:
        clipped = [
            LDUBox(b.xmin, b.ymin, max(b.zmin, z0), b.xmax, b.ymax, min(b.zmax, z1))
            for b in self.boxes
            if min(b.zmax, z1) > max(b.zmin, z0)
        ]
        return TargetShapeBoxes(boxes=clipped)


class SparseTiledTargetField:
    """
    Round 7 Critique 5D Fix:
    Sparse 3D Tiled Target Occupancy & Prefix-Sum Field with `uint32` accumulators.
    Partitions world space into sparse `160 x 160 x 192 LDU` tiles (`8 x 8` studs by `8` bricks
    = `80 x 80 x 96 = 614,400` cells at `2-LDU` resolution). Because `614,400 > 65,535`,
    per-tile prefix-sum accumulators use `np.uint32` (`accumulator_dtype == "uint32"`) to
    guarantee zero integer overflow across dense or full tiles while allocating memory only
    for non-empty tiles.
    """

    TILE_X_LDU: int = 160
    TILE_Y_LDU: int = 160
    TILE_Z_LDU: int = 192
    CELL_STEP_LDU: int = 2

    def __init__(self, boxes: Sequence[LDUBox]) -> None:
        self.boxes: Tuple[LDUBox, ...] = tuple(b for b in boxes if b.volume_ldu3 > 0)
        self.accumulator_dtype: str = "uint32"
        self._tile_prefix_sums: Dict[Tuple[int, int, int], np.ndarray] = {}
        self._tile_prefix_flat: Dict[Tuple[int, int, int], memoryview] = {}
        self._tile_boxes: Dict[Tuple[int, int, int], List[LDUBox]] = {}
        self._active_tiles: Set[Tuple[int, int, int]] = set()
        s = self.CELL_STEP_LDU
        self._all_even_aligned: bool = all(
            (b.xmin % s == 0)
            and (b.xmax % s == 0)
            and (b.ymin % s == 0)
            and (b.ymax % s == 0)
            and (b.zmin % s == 0)
            and (b.zmax % s == 0)
            for b in self.boxes
        )
        if self.boxes:
            self.xmin = min(b.xmin for b in self.boxes)
            self.ymin = min(b.ymin for b in self.boxes)
            self.zmin = min(b.zmin for b in self.boxes)
            self.xmax = max(b.xmax for b in self.boxes)
            self.ymax = max(b.ymax for b in self.boxes)
            self.zmax = max(b.zmax for b in self.boxes)
        else:
            self.xmin = self.ymin = self.zmin = 0
            self.xmax = self.ymax = self.zmax = 0

        for b in self.boxes:
            tx0 = b.xmin // self.TILE_X_LDU
            tx1 = (b.xmax - 1) // self.TILE_X_LDU
            ty0 = b.ymin // self.TILE_Y_LDU
            ty1 = (b.ymax - 1) // self.TILE_Y_LDU
            tz0 = b.zmin // self.TILE_Z_LDU
            tz1 = (b.zmax - 1) // self.TILE_Z_LDU
            for tx in range(tx0, tx1 + 1):
                for ty in range(ty0, ty1 + 1):
                    for tz in range(tz0, tz1 + 1):
                        tk = (tx, ty, tz)
                        self._active_tiles.add(tk)
                        self._tile_boxes.setdefault(tk, []).append(b)
        self._cached_total_volume_ldu3: Optional[int] = None

    @property
    def num_allocated_tiles(self) -> int:
        return len(self._active_tiles)

    @property
    def num_materialized_tiles(self) -> int:
        return len(self._tile_prefix_sums)

    def occupied_bounds_ldu(self) -> Tuple[int, int, int, int, int, int]:
        """Return `(xmin, ymin, zmin, xmax, ymax, zmax)` of occupied target geometry."""
        return (self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax)

    @property
    def total_volume_ldu3(self) -> int:
        if self._cached_total_volume_ldu3 is not None:
            return self._cached_total_volume_ldu3
        if not self.boxes:
            self._cached_total_volume_ldu3 = 0
            return 0
        if self._all_even_aligned:
            cell_vol = self.CELL_STEP_LDU ** 3  # 8 LDU^3
            tot = 0
            for tk in self._active_tiles:
                p = self.get_or_build_tile_prefix_u32(tk)
                tot += int(p[-1, -1, -1]) * cell_vol
            self._cached_total_volume_ldu3 = tot
            return tot
        self._cached_total_volume_ldu3 = sum(b.volume_ldu3 for b in self.boxes)
        return self._cached_total_volume_ldu3

    def get_or_build_tile_prefix_u32(self, tile_key: Tuple[int, int, int]) -> np.ndarray:
        """Lazily build the `uint32` 3D prefix sum array (`81 x 81 x 97`) for `tile_key`."""
        if tile_key in self._tile_prefix_sums:
            return self._tile_prefix_sums[tile_key]
        tx, ty, tz = tile_key
        ox = tx * self.TILE_X_LDU
        oy = ty * self.TILE_Y_LDU
        oz = tz * self.TILE_Z_LDU
        nx = self.TILE_X_LDU // self.CELL_STEP_LDU  # 80
        ny = self.TILE_Y_LDU // self.CELL_STEP_LDU  # 80
        nz = self.TILE_Z_LDU // self.CELL_STEP_LDU  # 96
        prefix = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.uint32)
        p_view = prefix[1:, 1:, 1:]
        s = self.CELL_STEP_LDU
        for b in self._tile_boxes.get(tile_key, ()):
            x0 = max(b.xmin, ox)
            x1 = min(b.xmax, ox + self.TILE_X_LDU)
            y0 = max(b.ymin, oy)
            y1 = min(b.ymax, oy + self.TILE_Y_LDU)
            z0 = max(b.zmin, oz)
            z1 = min(b.zmax, oz + self.TILE_Z_LDU)
            if x1 > x0 and y1 > y0 and z1 > z0:
                ix0, ix1 = (x0 - ox) // s, (x1 - ox) // s
                iy0, iy1 = (y0 - oy) // s, (y1 - oy) // s
                iz0, iz1 = (z0 - oz) // s, (z1 - oz) // s
                p_view[ix0:ix1, iy0:iy1, iz0:iz1] = 1
        np.cumsum(prefix, axis=0, out=prefix)
        np.cumsum(prefix, axis=1, out=prefix)
        np.cumsum(prefix, axis=2, out=prefix)
        self._tile_prefix_sums[tile_key] = prefix
        self._tile_prefix_flat[tile_key] = memoryview(prefix.ravel())
        return prefix

    def query_box_volume(self, qb: LDUBox) -> int:
        """
        Query exact integer LDU^3 intersection volume of `qb` directly against sparse `uint32`
        prefix-sum tiles (`get_or_build_tile_prefix_u32`), allocating ZERO global dense grids.
        """
        if not self.boxes:
            return 0
        qx0 = qb.xmin if qb.xmin > self.xmin else self.xmin
        qx1 = qb.xmax if qb.xmax < self.xmax else self.xmax
        if qx1 <= qx0:
            return 0
        qy0 = qb.ymin if qb.ymin > self.ymin else self.ymin
        qy1 = qb.ymax if qb.ymax < self.ymax else self.ymax
        if qy1 <= qy0:
            return 0
        qz0 = qb.zmin if qb.zmin > self.zmin else self.zmin
        qz1 = qb.zmax if qb.zmax < self.zmax else self.zmax
        if qz1 <= qz0:
            return 0

        s = self.CELL_STEP_LDU
        qb_even = (
            self._all_even_aligned
            and ((qx0 | qx1 | qy0 | qy1 | qz0 | qz1) & 1 == 0)
        )
        tx0 = qx0 // self.TILE_X_LDU
        tx1 = (qx1 - 1) // self.TILE_X_LDU
        ty0 = qy0 // self.TILE_Y_LDU
        ty1 = (qy1 - 1) // self.TILE_Y_LDU
        tz0 = qz0 // self.TILE_Z_LDU
        tz1 = (qz1 - 1) // self.TILE_Z_LDU

        if qb_even:
            # Strides for (81, 81, 97) uint32 prefix buffer: sz = 97, syz = 81 * 97 = 7857
            sz = (self.TILE_Z_LDU // s) + 1
            syz = ((self.TILE_Y_LDU // s) + 1) * sz
            total_cells = 0
            for tx in range(tx0, tx1 + 1):
                ox = tx * self.TILE_X_LDU
                lx0 = qx0 if qx0 > ox else ox
                lx1 = qx1 if qx1 < ox + self.TILE_X_LDU else ox + self.TILE_X_LDU
                ix0_s = ((lx0 - ox) >> 1) * syz
                ix1_s = ((lx1 - ox) >> 1) * syz
                for ty in range(ty0, ty1 + 1):
                    oy = ty * self.TILE_Y_LDU
                    ly0 = qy0 if qy0 > oy else oy
                    ly1 = qy1 if qy1 < oy + self.TILE_Y_LDU else oy + self.TILE_Y_LDU
                    iy0_s = ((ly0 - oy) >> 1) * sz
                    iy1_s = ((ly1 - oy) >> 1) * sz
                    for tz in range(tz0, tz1 + 1):
                        tk = (tx, ty, tz)
                        if tk not in self._active_tiles:
                            continue
                        oz = tz * self.TILE_Z_LDU
                        lz0 = qz0 if qz0 > oz else oz
                        lz1 = qz1 if qz1 < oz + self.TILE_Z_LDU else oz + self.TILE_Z_LDU
                        iz0 = (lz0 - oz) >> 1
                        iz1 = (lz1 - oz) >> 1
                        mv = self._tile_prefix_flat.get(tk)
                        if mv is None:
                            self.get_or_build_tile_prefix_u32(tk)
                            mv = self._tile_prefix_flat[tk]
                        c_tile = (
                            mv[ix1_s + iy1_s + iz1]
                            - mv[ix0_s + iy1_s + iz1]
                            - mv[ix1_s + iy0_s + iz1]
                            - mv[ix1_s + iy1_s + iz0]
                            + mv[ix0_s + iy0_s + iz1]
                            + mv[ix0_s + iy1_s + iz0]
                            + mv[ix1_s + iy0_s + iz0]
                            - mv[ix0_s + iy0_s + iz0]
                        )
                        total_cells += c_tile
            return total_cells << 3

        # Fallback for odd-LDU aligned boxes via tile-local box buckets (deduplicated by object id)
        seen_ids: Set[int] = set()
        vol = 0
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                for tz in range(tz0, tz1 + 1):
                    for tb in self._tile_boxes.get((tx, ty, tz), ()):
                        bid = id(tb)
                        if bid in seen_ids:
                            continue
                        seen_ids.add(bid)
                        ix = min(qb.xmax, tb.xmax) - max(qb.xmin, tb.xmin)
                        if ix <= 0:
                            continue
                        iy = min(qb.ymax, tb.ymax) - max(qb.ymin, tb.ymin)
                        if iy <= 0:
                            continue
                        iz = min(qb.zmax, tb.zmax) - max(qb.zmin, tb.zmin)
                        if iz <= 0:
                            continue
                        vol += ix * iy * iz
        return vol

    def band_volume_ldu3(self, z0: int, z1: int) -> int:
        """Return total target volume within vertical band `[z0, z1]` using sparse `uint32` tiles."""
        if not self.boxes or z1 <= z0:
            return 0
        return self.query_box_volume(
            LDUBox(self.xmin, self.ymin, z0, self.xmax, self.ymax, z1)
        )

    def slice_z_band(self, z0: int, z1: int) -> "ZBandTargetView":
        """Return a zero-copy `ZBandTargetView` restricted to `[z0, z1]` reusing cached `uint32` tiles."""
        return ZBandTargetView(parent=self, z0=z0, z1=z1)

    def intersection_volume(self, query_boxes: Sequence[LDUBox]) -> int:
        if len(query_boxes) == 1:
            return self.query_box_volume(query_boxes[0])
        return sum(self.query_box_volume(qb) for qb in query_boxes)

    def contains_point(self, pt: Tuple[int, int, int]) -> bool:
        x, y, z = pt
        if x <= self.xmin or x >= self.xmax or y <= self.ymin or y >= self.ymax or z <= self.zmin or z >= self.zmax:
            return False
        tk = (x // self.TILE_X_LDU, y // self.TILE_Y_LDU, z // self.TILE_Z_LDU)
        if tk not in self._active_tiles:
            return False
        if self._all_even_aligned:
            s = self.CELL_STEP_LDU
            ox, oy, oz = tk[0] * self.TILE_X_LDU, tk[1] * self.TILE_Y_LDU, tk[2] * self.TILE_Z_LDU
            ix, iy, iz = (x - ox) >> 1, (y - oy) >> 1, (z - oz) >> 1
            mv = self._tile_prefix_flat.get(tk)
            if mv is None:
                self.get_or_build_tile_prefix_u32(tk)
                mv = self._tile_prefix_flat[tk]
            sz = (self.TILE_Z_LDU // s) + 1
            syz = ((self.TILE_Y_LDU // s) + 1) * sz
            ix0_s, ix1_s = ix * syz, (ix + 1) * syz
            iy0_s, iy1_s = iy * sz, (iy + 1) * sz
            iz0, iz1 = iz, iz + 1
            occ_cell = (
                mv[ix1_s + iy1_s + iz1]
                - mv[ix0_s + iy1_s + iz1]
                - mv[ix1_s + iy0_s + iz1]
                - mv[ix1_s + iy1_s + iz0]
                + mv[ix0_s + iy0_s + iz1]
                + mv[ix0_s + iy1_s + iz0]
                + mv[ix1_s + iy0_s + iz0]
                - mv[ix0_s + iy0_s + iz0]
            )
            if occ_cell == 0:
                return False
            if (x % s != 0) and (y % s != 0) and (z % s != 0):
                return True
            # On an even 2-LDU plane: verify interior containment across the +/-1 LDU neighborhood
            # (strictly inside the union of occupied 2-LDU cells!) or within a single box
            if any(b.contains_point_strict(pt) for b in self._tile_boxes[tk]):
                return True
            # Also handle points sitting on an internal seam shared by two flush target boxes!
            for dx in (-1, 1) if (x % s == 0) else (0,):
                for dy in (-1, 1) if (y % s == 0) else (0,):
                    for dz in (-1, 1) if (z % s == 0) else (0,):
                        nx_p, ny_p, nz_p = x + dx, y + dy, z + dz
                        ntk = (nx_p // self.TILE_X_LDU, ny_p // self.TILE_Y_LDU, nz_p // self.TILE_Z_LDU)
                        if ntk not in self._active_tiles:
                            return False
                        nox, noy, noz = ntk[0] * self.TILE_X_LDU, ntk[1] * self.TILE_Y_LDU, ntk[2] * self.TILE_Z_LDU
                        nix, niy, niz = (nx_p - nox) // s, (ny_p - noy) // s, (nz_p - noz) // s
                        np_arr = self.get_or_build_tile_prefix_u32(ntk)
                        n_occ = (
                            int(np_arr[nix + 1, niy + 1, niz + 1])
                            - int(np_arr[nix, niy + 1, niz + 1])
                            - int(np_arr[nix + 1, niy, niz + 1])
                            - int(np_arr[nix + 1, niy + 1, niz])
                            + int(np_arr[nix, niy, niz + 1])
                            + int(np_arr[nix, niy + 1, niz])
                            + int(np_arr[nix + 1, niy, niz])
                            - int(np_arr[nix, niy, niz])
                        )
                        if n_occ == 0:
                            return False
            return True
        return any(b.contains_point_strict(pt) for b in self._tile_boxes[tk])

    def iou_with_placed_boxes(self, placed_boxes: Sequence[LDUBox]) -> float:
        tot = self.total_volume_ldu3
        if not placed_boxes and tot == 0:
            return 1.0
        placed_vol = sum(b.volume_ldu3 for b in placed_boxes)
        inter_vol = self.intersection_volume(placed_boxes)
        union_vol = tot + placed_vol - inter_vol
        return float(inter_vol) / float(max(union_vol, 1))

    def precision_of_placed_boxes(self, placed_boxes: Sequence[LDUBox]) -> float:
        if not placed_boxes:
            return 1.0
        placed_vol = sum(b.volume_ldu3 for b in placed_boxes)
        inter_vol = self.intersection_volume(placed_boxes)
        return float(inter_vol) / float(max(placed_vol, 1))


class ZBandTargetView:
    """
    Zero-copy vertical Z-band view `[z0, z1]` over a `SparseTiledTargetField` (WS-1A/1B).
    Reuses the parent field's materialized `uint32` 3D prefix-sum tiles in O(1) without
    allocating or recomputing per-course tile arrays.
    """

    __slots__ = ("parent", "z0", "z1", "_cached_vol")

    def __init__(self, parent: SparseTiledTargetField, z0: int, z1: int) -> None:
        self.parent = parent
        self.z0 = int(z0)
        self.z1 = int(z1)
        self._cached_vol: Optional[int] = None

    @property
    def total_volume_ldu3(self) -> int:
        if self._cached_vol is None:
            self._cached_vol = self.parent.band_volume_ldu3(self.z0, self.z1)
        return self._cached_vol

    def occupied_bounds_ldu(self) -> Tuple[int, int, int, int, int, int]:
        xmin, ymin, zmin, xmax, ymax, zmax = self.parent.occupied_bounds_ldu()
        return (xmin, ymin, max(zmin, self.z0), xmax, ymax, min(zmax, self.z1))

    def query_box_volume(self, qb: LDUBox) -> int:
        cz0 = max(qb.zmin, self.z0)
        cz1 = min(qb.zmax, self.z1)
        if cz1 <= cz0:
            return 0
        if cz0 == qb.zmin and cz1 == qb.zmax:
            return self.parent.query_box_volume(qb)
        return self.parent.query_box_volume(
            LDUBox(qb.xmin, qb.ymin, cz0, qb.xmax, qb.ymax, cz1)
        )

    def intersection_volume(self, query_boxes: Sequence[LDUBox]) -> int:
        if len(query_boxes) == 1:
            return self.query_box_volume(query_boxes[0])
        return sum(self.query_box_volume(qb) for qb in query_boxes)

    def contains_point(self, pt: Tuple[int, int, int]) -> bool:
        if pt[2] <= self.z0 or pt[2] >= self.z1:
            return False
        return self.parent.contains_point(pt)

    def iou_with_placed_boxes(self, placed_boxes: Sequence[LDUBox]) -> float:
        tot = self.total_volume_ldu3
        if not placed_boxes and tot == 0:
            return 1.0
        placed_vol = sum(b.volume_ldu3 for b in placed_boxes)
        inter_vol = self.intersection_volume(placed_boxes)
        union_vol = tot + placed_vol - inter_vol
        return float(inter_vol) / float(max(union_vol, 1))

    def precision_of_placed_boxes(self, placed_boxes: Sequence[LDUBox]) -> float:
        if not placed_boxes:
            return 1.0
        placed_vol = sum(b.volume_ldu3 for b in placed_boxes)
        inter_vol = self.intersection_volume(placed_boxes)
        return float(inter_vol) / float(max(placed_vol, 1))


@dataclass(frozen=True, slots=True)
class FrontierGraphBatch:
    """
    Complete topographical tensor representation of an Active Connection Frontier F_t
    and its verified Course Candidate Pool {c_1, ..., c_M}.
    """

    port_features: np.ndarray  # [N_ports, 24] float32
    edge_index: np.ndarray  # [2, E] int64 (directed edges u -> v)
    edge_rel_rot_id: np.ndarray  # [E] int64 in 0..23 (ΔR_uv)
    edge_rel_pos_ldu: np.ndarray  # [E, 3] float32 (Δr_uv scaled by 1/20 LDU)
    edge_attr: np.ndarray  # [E, 6] float32 (coplanar, snot_corner, shared_cand, seam_bridge, dist, dz)
    candidate_features: np.ndarray  # [M, 18] float32
    candidate_port_indices: Tuple[Tuple[int, ...], ...]  # for each candidate i, indices in 0..N_ports-1


def _compute_grounded_support_xy_bounds(state: AssemblyState) -> Optional[Tuple[int, int, int, int]]:
    """Return (xmin, ymin, xmax, ymax) of all grounded parts in world LDU coordinates."""
    xs_min: List[int] = []
    ys_min: List[int] = []
    xs_max: List[int] = []
    ys_max: List[int] = []
    for placed in state.placed_parts.values():
        if placed.is_grounded:
            for box in placed.world_boxes:
                xs_min.append(box.xmin)
                ys_min.append(box.ymin)
                xs_max.append(box.xmax)
                ys_max.append(box.ymax)
    if not xs_min:
        return None
    return (min(xs_min), min(ys_min), max(xs_max), max(ys_max))


def _xy_overhang_distance_ldu(
    pt: Tuple[int, int, int], support_bounds: Optional[Tuple[int, int, int, int]]
) -> float:
    """Compute horizontal overhang distance (in LDU) outside the grounded support XY box."""
    if support_bounds is None:
        return 100.0
    xmin, ymin, xmax, ymax = support_bounds
    dx = max(0, xmin - pt[0], pt[0] - xmax)
    dy = max(0, ymin - pt[1], pt[1] - ymax)
    return math.hypot(dx, dy)


def _compute_part_biconnected_ratios(
    state: AssemblyState,
    frontier: Optional[ActiveConnectionFrontier] = None,
) -> Dict[int, float]:
    """
    For each placed part, compute the fraction of its connected neighbor parts (within 2 hops)
    that are biconnected with it (distinguishing bridged seams from unbridged vertical seams).
    """
    parts = state.placed_parts
    ratios: Dict[int, float] = {}
    if len(parts) <= 20:
        for inst_a in parts:
            other_insts = [inst_b for inst_b in parts if inst_b != inst_a]
            if not other_insts:
                ratios[inst_a] = 1.0
                continue
            biconn_count = sum(
                1 for inst_b in other_insts if state.are_parts_biconnected(inst_a, inst_b)
            )
            ratios[inst_a] = float(biconn_count) / float(len(other_insts))
        return ratios

    # For large multi-course assemblies (> 20 parts), evaluate only parts owning active frontier
    # ports against their 2-hop contact-graph neighborhood in O(|F_t| * deg^2) time.
    active_owners = (
        {p.global_port_id[0] for p in frontier.ports}
        if frontier is not None
        else set(parts.keys())
    )
    part_adj = state._part_adj
    for inst_a in active_owners:
        nbrs_1 = part_adj.get(inst_a, set())
        nbrs_2: Set[int] = set(nbrs_1)
        for n1 in nbrs_1:
            nbrs_2.update(part_adj.get(n1, set()))
        nbrs_2.discard(inst_a)
        if not nbrs_2:
            ratios[inst_a] = 1.0
            continue
        biconn_count = sum(
            1 for inst_b in nbrs_2 if state.are_parts_biconnected(inst_a, inst_b)
        )
        ratios[inst_a] = float(biconn_count) / float(len(nbrs_2))
    return ratios


def build_frontier_and_candidate_tensors(
    state: AssemblyState,
    frontier: Optional[ActiveConnectionFrontier] = None,
    pool: Optional[CandidatePool] = None,
    target_shape: Optional[TargetShapeBoxes] = None,
) -> FrontierGraphBatch:
    """
    Construct the 24-channel Topographical Frontier Port Tensor, the Sparse Relative-Frame
    Edge Tensor, and the 18-channel Candidate Feature Tensor from `state`.

    Port Tensor Channels (24):
      0: polarity (+1 stud, -1 anti-stud)
      1: is_reachable (1.0 if REACHABLE, 0.0 if EXPOSED_BLOCKED)
      2..4: local gravity vector g_u = R_u^T g_world (gx, gy, gz)
      5: exact 1x1 feasibility (3024/3005 valid at port u)
      6..9: exact 1x2/1x4 directional feasibility along u's local (+X, -X, +Y, -Y)
      10: exact 2x2/2x4 quadrant feasibility ratio in [0, 1]
      11: exact SNOT primitive feasibility indicator (87087, 4070, 99780, 99207)
      12: log-normalized valid candidate degree at port u
      13: normalized topological support depth to ground
      14: owning part is_grounded indicator
      15: seam-bridge demand (fraction of nearby frontier ports on non-biconnected parts)
      16: owning part biconnected ratio in persistent ContactGraph
      17: horizontal cantilever overhang moment arm (in units of 5 studs = 100 LDU)
      18: normalized cumulative assembly mass per grounded support port
      19: lateral SNOT shear load indicator (1 - |g_{u,z}|)
      20: target-shape candidate precision at port u
      21: max target-shape volume gain at port u (normalized by 2x4 brick volume)
      22: local target-shape ray depth along outward normal (up to 4 brick heights)
      23: target-shape surface boundary proximity indicator
    """
    if frontier is None:
        frontier = state.get_active_frontier()
    if pool is None:
        pool = state.generate_candidate_pool(frontier=frontier)

    n_ports = len(frontier)
    m_cands = len(pool)
    port_feats = np.zeros((n_ports, FRONTIER_PORT_CHANNELS), dtype=np.float32)

    placed_parts = state.placed_parts
    support_bounds = _compute_grounded_support_xy_bounds(state)
    biconn_ratios = _compute_part_biconnected_ratios(state, frontier=frontier)
    total_mass = sum(
        get_brick_primitive(p.part_id).mass_grams for p in placed_parts.values()
    )
    grounded_count = max(1, sum(1 for p in placed_parts.values() if p.is_grounded))
    norm_mass_load = min(total_mass / (grounded_count * 10.0), 5.0)
    ref_2x4_vol = float(2 * STUD_PITCH_LDU * 4 * STUD_PITCH_LDU * BRICK_HEIGHT_LDU)

    # Pre-compute target overlap for every candidate
    cand_target_vols = np.zeros(m_cands, dtype=np.float32)
    cand_total_vols = np.zeros(m_cands, dtype=np.float32)
    for cid, cand in enumerate(pool.candidates):
        tot_v = float(sum(b.volume_ldu3 for b in cand.world_boxes))
        cand_total_vols[cid] = max(tot_v, 1.0)
        if target_shape is not None:
            cand_target_vols[cid] = float(target_shape.intersection_volume(cand.world_boxes))

    # Populate 24-channel frontier port features
    for u_idx, exp_u in enumerate(frontier.ports):
        gpid = exp_u.global_port_id
        inst_u = gpid[0]
        placed_u = placed_parts[inst_u]
        t_u = exp_u.world_frame
        cands_at_u = pool.port_to_candidates.get(gpid, ())

        # Ch 0..4: Polarity, Reachability, Local Gravity
        port_feats[u_idx, 0] = float(exp_u.polarity)
        port_feats[u_idx, 1] = 1.0 if exp_u.status.is_reachable else 0.0
        port_feats[u_idx, 2] = float(exp_u.gravity_local[0])
        port_feats[u_idx, 3] = float(exp_u.gravity_local[1])
        port_feats[u_idx, 4] = float(exp_u.gravity_local[2])

        # Ch 5..12: Exact Whole-Hull Feasibility from verified CandidatePool
        has_1x1 = 0.0
        dir_feas = [0.0, 0.0, 0.0, 0.0]  # +X, -X, +Y, -Y in u's local frame
        quad_covered: Set[Tuple[int, int]] = set()
        has_snot = 0.0

        for cid in cands_at_u:
            cand = pool.candidates[cid]
            prim = get_brick_primitive(cand.part_id)
            if cand.part_id in ("3024", "3005"):
                has_1x1 = 1.0
            if prim.is_snot:
                has_snot = 1.0

            # Measure candidate box center offset in u's local frame
            for wbox in cand.world_boxes:
                cx = (wbox.xmin + wbox.xmax) // 2
                cy = (wbox.ymin + wbox.ymax) // 2
                cz = (wbox.zmin + wbox.zmax) // 2
                rel_pt = t_u.inverse().transform_point((cx, cy, cz))
                rx, ry, _ = rel_pt
                if rx >= 8 and abs(ry) <= 5:
                    dir_feas[0] = 1.0
                if rx <= -8 and abs(ry) <= 5:
                    dir_feas[1] = 1.0
                if ry >= 8 and abs(rx) <= 5:
                    dir_feas[2] = 1.0
                if ry <= -8 and abs(rx) <= 5:
                    dir_feas[3] = 1.0
                if abs(rx) >= 8 and abs(ry) >= 8:
                    quad_covered.add((1 if rx > 0 else -1, 1 if ry > 0 else -1))

        port_feats[u_idx, 5] = has_1x1
        port_feats[u_idx, 6] = dir_feas[0]
        port_feats[u_idx, 7] = dir_feas[1]
        port_feats[u_idx, 8] = dir_feas[2]
        port_feats[u_idx, 9] = dir_feas[3]
        port_feats[u_idx, 10] = float(len(quad_covered)) / 4.0
        port_feats[u_idx, 11] = has_snot
        port_feats[u_idx, 12] = math.log1p(len(cands_at_u)) / math.log(64.0)

        # Ch 13..16: Structural Seam & Biconnected Topology
        port_feats[u_idx, 13] = min(float(exp_u.support_depth), 20.0) / 20.0
        port_feats[u_idx, 14] = 1.0 if placed_u.is_grounded else 0.0

        nearby_ports = 0
        unbraced_seam_ports = 0
        for v_idx, exp_v in enumerate(frontier.ports):
            if u_idx == v_idx:
                continue
            rel_uv = relative_transform(t_u, exp_v.world_frame)
            dist_ldu = math.sqrt(
                rel_uv.pos[0] ** 2 + rel_uv.pos[1] ** 2 + rel_uv.pos[2] ** 2
            )
            if dist_ldu <= 45.0:
                nearby_ports += 1
                inst_v = exp_v.global_port_id[0]
                if inst_v != inst_u and not state.are_parts_biconnected(inst_u, inst_v):
                    unbraced_seam_ports += 1

        port_feats[u_idx, 15] = (
            float(unbraced_seam_ports) / float(nearby_ports) if nearby_ports > 0 else 0.0
        )
        port_feats[u_idx, 16] = biconn_ratios.get(inst_u, 1.0)

        # Ch 17..19: Static Cantilever & Shear Load Proxy
        overhang_ldu = _xy_overhang_distance_ldu(t_u.pos, support_bounds)
        port_feats[u_idx, 17] = min(overhang_ldu / 100.0, 2.0)
        port_feats[u_idx, 18] = norm_mass_load
        port_feats[u_idx, 19] = 1.0 - abs(float(exp_u.gravity_local[2]))

        # Ch 20..23: Target-Shape Conditioning
        if target_shape is not None:
            if cands_at_u:
                precisions = [
                    cand_target_vols[cid] / cand_total_vols[cid] for cid in cands_at_u
                ]
                gains = [cand_target_vols[cid] / ref_2x4_vol for cid in cands_at_u]
                port_feats[u_idx, 20] = float(max(precisions))
                port_feats[u_idx, 21] = min(float(max(gains)), 2.0)

            step_dir = 1 if exp_u.polarity == PortPolarity.STUD else -1
            hits = 0
            for layer_k in range(1, 5):
                probe = t_u.transform_point((0, 0, step_dir * (layer_k * BRICK_HEIGHT_LDU - 12)))
                if target_shape.contains_point(probe):
                    hits += 1
            port_feats[u_idx, 22] = float(hits) / 4.0

            # Surface boundary indicator: inside target at +4 LDU but outside at +28 LDU
            p_near = t_u.transform_point((0, 0, step_dir * 4))
            p_far = t_u.transform_point((0, 0, step_dir * 28))
            port_feats[u_idx, 23] = (
                1.0
                if (target_shape.contains_point(p_near) and not target_shape.contains_point(p_far))
                else 0.0
            )

    # Build Sparse Relative-Frame Frontier Graph E_t
    # Co-consumed candidate lookup
    co_consumed_pairs: Set[Tuple[int, int]] = set()
    cand_port_idx_tuples: List[Tuple[int, ...]] = []
    for cand in pool.candidates:
        p_indices = tuple(
            sorted(
                frontier.port_index_by_id[gpid]
                for gpid in cand.consumed_ports
                if gpid in frontier.port_index_by_id
            )
        )
        cand_port_idx_tuples.append(p_indices)
        for i in range(len(p_indices)):
            for j in range(i + 1, len(p_indices)):
                u_i, v_i = p_indices[i], p_indices[j]
                co_consumed_pairs.add((u_i, v_i))
                co_consumed_pairs.add((v_i, u_i))

    src_list: List[int] = []
    dst_list: List[int] = []
    rel_rot_list: List[int] = []
    rel_pos_list: List[Tuple[float, float, float]] = []
    edge_attr_list: List[Tuple[float, float, float, float, float, float]] = []

    for u_idx in range(n_ports):
        exp_u = frontier.ports[u_idx]
        inst_u = exp_u.global_port_id[0]
        for v_idx in range(n_ports):
            exp_v = frontier.ports[v_idx]
            inst_v = exp_v.global_port_id[0]
            rel_uv = relative_transform(exp_u.world_frame, exp_v.world_frame)
            dx, dy, dz = rel_uv.pos
            linf_xy = max(abs(dx), abs(dy))
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

            is_coplanar = (rel_uv.rot_id in C4_Z_ROT_IDS) and (abs(dz) <= 8) and (linf_xy <= 160)
            is_snot_corner = (rel_uv.rot_id not in C4_Z_ROT_IDS) and (dist_3d <= 60.0)
            is_shared_cand = (u_idx, v_idx) in co_consumed_pairs
            is_seam_edge = (
                (inst_u != inst_v)
                and (dist_3d <= 45.0)
                and not state.are_parts_biconnected(inst_u, inst_v)
            )

            if u_idx == v_idx or is_coplanar or is_snot_corner or is_shared_cand or is_seam_edge:
                src_list.append(u_idx)
                dst_list.append(v_idx)
                rel_rot_list.append(rel_uv.rot_id)
                rel_pos_list.append(
                    (
                        float(dx) / float(STUD_PITCH_LDU),
                        float(dy) / float(STUD_PITCH_LDU),
                        float(dz) / float(STUD_PITCH_LDU),
                    )
                )
                edge_attr_list.append(
                    (
                        1.0 if is_coplanar else 0.0,
                        1.0 if is_snot_corner else 0.0,
                        1.0 if is_shared_cand else 0.0,
                        1.0 if is_seam_edge else 0.0,
                        dist_3d / 160.0,
                        float(dz) / float(BRICK_HEIGHT_LDU),
                    )
                )

    edge_index = (
        np.array([src_list, dst_list], dtype=np.int64)
        if src_list
        else np.zeros((2, 0), dtype=np.int64)
    )
    edge_rel_rot_id = np.array(rel_rot_list, dtype=np.int64)
    edge_rel_pos_ldu = (
        np.array(rel_pos_list, dtype=np.float32)
        if rel_pos_list
        else np.zeros((0, 3), dtype=np.float32)
    )
    edge_attr = (
        np.array(edge_attr_list, dtype=np.float32)
        if edge_attr_list
        else np.zeros((0, 6), dtype=np.float32)
    )

    # Populate 18-channel Candidate Feature Tensor
    cand_feats = np.zeros((m_cands, CANDIDATE_FEATURE_CHANNELS), dtype=np.float32)
    part_id_list = sorted(BRICK_CATALOG)
    part_type_norm = {pid: idx / max(1.0, float(len(part_id_list) - 1)) for idx, pid in enumerate(part_id_list)}

    for cid, cand in enumerate(pool.candidates):
        prim = get_brick_primitive(cand.part_id)
        p_indices = cand_port_idx_tuples[cid]
        tot_v = cand_total_vols[cid]
        targ_v = cand_target_vols[cid]
        c_box0 = cand.world_boxes[0]
        cx = (c_box0.xmin + c_box0.xmax) // 2
        cy = (c_box0.ymin + c_box0.ymax) // 2
        cz = (c_box0.zmin + c_box0.zmax) // 2

        cand_feats[cid, 0] = part_type_norm.get(cand.part_id, 0.5)
        cand_feats[cid, 1] = tot_v / ref_2x4_vol
        cand_feats[cid, 2] = prim.mass_grams / 5.0
        cand_feats[cid, 3] = 1.0 if prim.is_snot else 0.0
        cand_feats[cid, 4] = float(len(cand.consumed_ports)) / 8.0
        cand_feats[cid, 5] = float(len(cand.bridged_instances)) / 4.0
        cand_feats[cid, 6] = 1.0 if cand.closes_structural_seam else 0.0
        cand_feats[cid, 7] = 1.0 if cand.creates_biconnected_cycle else 0.0
        cand_feats[cid, 8] = float(cand.gravity_local[0])
        cand_feats[cid, 9] = float(cand.gravity_local[1])
        cand_feats[cid, 10] = float(cand.gravity_local[2])
        cand_feats[cid, 11] = min(_xy_overhang_distance_ldu((cx, cy, cz), support_bounds) / 100.0, 2.0)
        cand_feats[cid, 12] = math.log1p(len(pool.conflict_adjacency[cid])) / math.log(128.0)
        cand_feats[cid, 13] = targ_v / tot_v if target_shape is not None else 0.0
        cand_feats[cid, 14] = targ_v / ref_2x4_vol if target_shape is not None else 0.0
        cand_feats[cid, 15] = (tot_v - targ_v) / ref_2x4_vol if target_shape is not None else 0.0
        if p_indices:
            cand_feats[cid, 16] = float(np.mean(port_feats[list(p_indices), 13]))  # support depth
            cand_feats[cid, 17] = float(np.mean(port_feats[list(p_indices), 15]))  # seam demand

    return FrontierGraphBatch(
        port_features=port_feats,
        edge_index=edge_index,
        edge_rel_rot_id=edge_rel_rot_id,
        edge_rel_pos_ldu=edge_rel_pos_ldu,
        edge_attr=edge_attr,
        candidate_features=cand_feats,
        candidate_port_indices=tuple(cand_port_idx_tuples),
    )
