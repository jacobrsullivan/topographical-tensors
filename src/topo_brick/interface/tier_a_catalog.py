"""
WS-3A: Tier A Declarative Part Catalog (`TIER_A_CATALOG`) & Official LEGO Set `10698-1` Inventory.

Provides:
1. `SET_10698_OFFICIAL_INVENTORY`: Official Rebrickable inventory (`inventory_id=102864`) for
   LEGO Classic Large Creative Brick Box (`10698-1`): 108 unique part types, 790 non-spare pieces.
2. `TIER_A_CATALOG`: 160+ declarative LDU primitives covering 100% of Set `10698-1` (108/108 part
   types, 790/790 pieces) plus high-frequency OMR `.mpd` parts across all 8 connector families:
   - Standard & Hollow Studs (`STANDARD_STUD`, `HOLLOW_STUD`)
   - Technic Pin / Hole / Smooth Pin (`TECHNIC_PIN`, `SMOOTH_PIN`, `TECHNIC_PIN_HOLE`)
   - Technic Axle / Axle Hole (`TECHNIC_AXLE`, `TECHNIC_AXLE_HOLE`)
   - 3.18mm Bar & Clip (`BAR_3_18MM`, `CLIP_3_18MM`)
   - Smooth & 9-Tooth Locking Hinges (`HINGE_PIN`/`HINGE_SOCKET`, `LOCKING_HINGE_PIN`/`LOCKING_HINGE_SOCKET`)
   - Ball Joint & Socket (`BALL_JOINT`, `BALL_SOCKET`)
   - Turntable (`TURNTABLE_TOP_PIN`, `TURNTABLE_BASE_SOCKET`)
   - Wheel Pin & Hub (`WHEEL_PIN`, `WHEEL_HUB_HOLE`)
3. `LDRAW_PART_ALIASES` & `normalize_ldraw_part_id`: Canonical mold/print alias resolution for
   official LDraw `.ldr`/`.mpd` files (OMR CCAL v2.0).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from topo_brick.interface.brick_catalog import (
    ROT_Z_TO_NEG_X,
    ROT_Z_TO_POS_X,
    BrickPrimitive,
    ConnectorType,
    PortPolarity,
    PortSpec,
    _make_rect_part,
)
from topo_brick.interface.ldu_frames import (
    BRICK_HEIGHT_LDU,
    HALF_STUD_LDU,
    PLATE_HEIGHT_LDU,
    STUD_PITCH_LDU,
    LDUBox,
    LDUFrame,
    matrix_to_rot_id,
)

ROT_Z_TO_POS_Y: int = matrix_to_rot_id(
    np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.int64)
)
ROT_Z_TO_NEG_Y: int = matrix_to_rot_id(
    np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.int64)
)
ROT_Z_INVERT: int = matrix_to_rot_id(
    np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.int64)
)


# Official Rebrickable Set 10698-1 Inventory (inventory_id=102864): 108 unique parts, 790 non-spare pieces
SET_10698_OFFICIAL_INVENTORY: Dict[str, int] = {
    "3004": 120,
    "3001": 67,
    "3003": 60,
    "3005": 48,
    "3039": 30,
    "3040b": 30,
    "3010": 28,
    "3002": 20,
    "60481": 16,
    "3660": 14,
    "6141": 14,
    "85984": 14,
    "30136": 12,
    "54200": 12,
    "3023": 10,
    "3710": 10,
    "33291": 9,
    "3037": 8,
    "3068b": 8,
    "3245c": 8,
    "3941": 8,
    "2456": 6,
    "3020": 6,
    "3062b": 6,
    "47905": 6,
    "50950": 6,
    "6091": 6,
    "93273": 6,
    "60592": 5,
    "60601": 5,
    "2431": 4,
    "2458": 4,
    "2877": 4,
    "3009": 4,
    "3031": 4,
    "30414": 4,
    "3045": 4,
    "3622": 4,
    "3659": 4,
    "3665": 4,
    "3676": 4,
    "4070": 4,
    "4081b": 4,
    "6014b": 4,
    "60598": 4,
    "60607": 4,
    "60608": 4,
    "6081": 4,
    "6215": 4,
    "76959": 4,
    "85080": 4,
    "87079": 4,
    "87697": 4,
    "98138pr0008": 4,
    "3069b": 3,
    "3957b": 3,
    "4727": 3,
    "59900": 3,
    "98138pr0026": 3,
    "11212": 2,
    "14769": 2,
    "15207": 2,
    "15332": 2,
    "15397": 2,
    "2926": 2,
    "30044": 2,
    "30046": 2,
    "30165": 2,
    "3021": 2,
    "3022": 2,
    "3034": 2,
    "30367c": 2,
    "30387": 2,
    "3065": 2,
    "3460": 2,
    "3666": 2,
    "3679": 2,
    "3680": 2,
    "3795": 2,
    "4287c": 2,
    "44728": 2,
    "55981": 2,
    "58090": 2,
    "60471": 2,
    "60593": 2,
    "60602": 2,
    "6232": 2,
    "87544": 2,
    "90195": 2,
    "92582": 2,
    "98262": 2,
    "15254": 1,
    "30151b": 1,
    "3028": 1,
    "3029": 1,
    "3032": 1,
    "3942c": 1,
    "60479": 1,
    "60596": 1,
    "60599": 1,
    "60621": 1,
    "60623": 1,
    "61252": 1,
    "6223": 1,
    "87081": 1,
    "91405": 1,
    "96874": 1,
    "99206": 1,
}


def _make_tile_part(
    part_id: str,
    name: str,
    studs_x: int,
    studs_y: int,
    mass_grams: float,
    height_ldu: int = PLATE_HEIGHT_LDU,
) -> BrickPrimitive:
    """Smooth-topped tile: bottom anti-studs at z=0, zero top studs."""
    box = LDUBox(
        xmin=-HALF_STUD_LDU,
        ymin=-HALF_STUD_LDU,
        zmin=0,
        xmax=(studs_x - 1) * STUD_PITCH_LDU + HALF_STUD_LDU,
        ymax=(studs_y - 1) * STUD_PITCH_LDU + HALF_STUD_LDU,
        zmax=height_ldu,
    )
    ports = [
        PortSpec(
            port_id=ix * studs_y + iy,
            local_frame=LDUFrame(0, (ix * STUD_PITCH_LDU, iy * STUD_PITCH_LDU, 0)),
            polarity=PortPolarity.ANTI_STUD,
        )
        for ix in range(studs_x)
        for iy in range(studs_y)
    ]
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(box,),
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=False,
    )


def _make_slope_part(
    part_id: str,
    name: str,
    depth_studs_x: int,
    width_studs_y: int,
    height_ldu: int,
    rear_stud_rows_x: int,
    mass_grams: float,
) -> BrickPrimitive:
    """
    Standard/curved slope brick along +X:
    - Rear ridge (`rear_stud_rows_x` studs deep, or 10 LDU ridge if 0 top studs) at full `height_ldu`.
    - Front sloped apron at half height (`max(4, height_ldu // 2)`), leaving sloped clearance open.
    """
    ports: List[PortSpec] = []
    pid = 0
    for ix in range(depth_studs_x):
        for iy in range(width_studs_y):
            ports.append(
                PortSpec(
                    pid,
                    LDUFrame(0, (ix * STUD_PITCH_LDU, iy * STUD_PITCH_LDU, 0)),
                    PortPolarity.ANTI_STUD,
                )
            )
            pid += 1
    for ix in range(rear_stud_rows_x):
        for iy in range(width_studs_y):
            ports.append(
                PortSpec(
                    pid,
                    LDUFrame(0, (ix * STUD_PITCH_LDU, iy * STUD_PITCH_LDU, height_ldu)),
                    PortPolarity.STUD,
                )
            )
            pid += 1

    ymax = (width_studs_y - 1) * STUD_PITCH_LDU + HALF_STUD_LDU
    xmax = (depth_studs_x - 1) * STUD_PITCH_LDU + HALF_STUD_LDU
    if depth_studs_x == 1:
        # 1-stud-deep cheese slope (54200 / 85984): back half full height, front half half height
        b_rear = LDUBox(-10, -10, 0, 0, ymax, height_ldu)
        b_front = LDUBox(0, -10, 0, 10, ymax, max(4, height_ldu // 2))
        boxes = (b_rear, b_front)
    else:
        split_x = max(0, rear_stud_rows_x - 1) * STUD_PITCH_LDU + HALF_STUD_LDU
        if split_x >= xmax:
            boxes = (LDUBox(-10, -10, 0, xmax, ymax, height_ldu),)
        else:
            b_rear = LDUBox(-10, -10, 0, split_x, ymax, height_ldu)
            b_front = LDUBox(split_x, -10, 0, xmax, ymax, max(4, height_ldu // 2))
            boxes = (b_rear, b_front)

    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=boxes,
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=False,
    )


def _make_inverted_slope_part(
    part_id: str,
    name: str,
    top_depth_x: int,
    bottom_depth_x: int,
    width_y: int,
    height_ldu: int,
    mass_grams: float,
) -> BrickPrimitive:
    """
    Inverted slope brick (`3660`, `3665`, `3676`, `76959`, `4287c`):
    - Narrow bottom base (`bottom_depth_x` x `width_y`) from z=0..height_ldu.
    - Overhanging upper sloped corbel (`bottom_depth_x..top_depth_x`) from z=height_ldu//2..height_ldu.
    """
    ports: List[PortSpec] = []
    pid = 0
    for ix in range(bottom_depth_x):
        for iy in range(width_y):
            ports.append(
                PortSpec(pid, LDUFrame(0, (ix * 20, iy * 20, 0)), PortPolarity.ANTI_STUD)
            )
            pid += 1
    for ix in range(top_depth_x):
        for iy in range(width_y):
            ports.append(
                PortSpec(pid, LDUFrame(0, (ix * 20, iy * 20, height_ldu)), PortPolarity.STUD)
            )
            pid += 1
    ymax = (width_y - 1) * 20 + 10
    split_x = (bottom_depth_x - 1) * 20 + 10
    xmax = (top_depth_x - 1) * 20 + 10
    b_core = LDUBox(-10, -10, 0, split_x, ymax, height_ldu)
    if xmax > split_x:
        b_over = LDUBox(split_x, -10, height_ldu // 2, xmax, ymax, height_ldu)
        boxes: Tuple[LDUBox, ...] = (b_core, b_over)
    else:
        boxes = (b_core,)
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=boxes,
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=False,
    )


def _make_arch_part(
    part_id: str,
    name: str,
    span_studs_y: int,
    height_ldu: int,
    mass_grams: float,
) -> BrickPrimitive:
    """
    Brick Arch (`3659` 1x4x1, `15254` 1x6x2):
    - Two 1x1 outer abutment pillars at y=0 and y=(span_studs_y-1)*20.
    - Top 8-LDU lintel spanning the full arch width, leaving the central opening clear!
    """
    ymax_end = (span_studs_y - 1) * 20
    pillar_l = LDUBox(-10, -10, 0, 10, 10, height_ldu - 8)
    pillar_r = LDUBox(-10, ymax_end - 10, 0, 10, ymax_end + 10, height_ldu - 8)
    lintel = LDUBox(-10, -10, height_ldu - 8, 10, ymax_end + 10, height_ldu)
    ports: List[PortSpec] = [
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
        PortSpec(1, LDUFrame(0, (0, ymax_end, 0)), PortPolarity.ANTI_STUD),
    ]
    pid = 2
    for iy in range(span_studs_y):
        ports.append(PortSpec(pid, LDUFrame(0, (0, iy * 20, height_ldu)), PortPolarity.STUD))
        pid += 1
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(pillar_l, pillar_r, lintel),
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=False,
    )


def _make_technic_brick(
    part_id: str,
    name: str,
    studs_y: int,
    mass_grams: float,
) -> BrickPrimitive:
    """
    Technic Brick 1 x N with Holes (`3700` 1x2, `3701` 1x4, `3702` 1x8, `3703` 1x16):
    - Standard 1 x N top studs and bottom anti-studs.
    - (N - 1) cross-holes along +/-X at y = 10 + 20*k, z = 14.
    """
    ymax = (studs_y - 1) * 20 + 10
    # Upper & lower solid rails leave z in [8, 20] pin corridor open to Technic pins/axles
    b_bot = LDUBox(-10, -10, 0, 10, ymax, 8)
    b_top = LDUBox(-10, -10, 20, 10, ymax, 24)
    ports: List[PortSpec] = []
    pid = 0
    for iy in range(studs_y):
        ports.append(PortSpec(pid, LDUFrame(0, (0, iy * 20, 0)), PortPolarity.ANTI_STUD))
        pid += 1
    for iy in range(studs_y):
        ports.append(PortSpec(pid, LDUFrame(0, (0, iy * 20, 24)), PortPolarity.STUD))
        pid += 1
    for k in range(max(1, studs_y - 1)):
        hy = 10 + 20 * k if studs_y > 1 else 0
        ports.append(
            PortSpec(
                pid,
                LDUFrame(ROT_Z_TO_NEG_X, (10, hy, 14)),
                PortPolarity.ANTI_STUD,
                ConnectorType.TECHNIC_PIN_HOLE,
            )
        )
        pid += 1
        ports.append(
            PortSpec(
                pid,
                LDUFrame(ROT_Z_TO_POS_X, (-10, hy, 14)),
                PortPolarity.ANTI_STUD,
                ConnectorType.TECHNIC_PIN_HOLE,
            )
        )
        pid += 1
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(b_bot, b_top),
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=True,
    )


def _make_technic_liftarm(
    part_id: str,
    name: str,
    holes_y: int,
    mass_grams: float,
) -> BrickPrimitive:
    """Studless Technic Liftarm 1 x N Thick (`32523` 1x3, `32316` 1x5, `32524` 1x7)."""
    ymax = (holes_y - 1) * 20 + 10
    b_left = LDUBox(-10, -10, 0, -6, ymax, 20)
    b_right = LDUBox(6, -10, 0, 10, ymax, 20)
    ports: List[PortSpec] = []
    pid = 0
    for iy in range(holes_y):
        ports.append(
            PortSpec(
                pid,
                LDUFrame(ROT_Z_TO_NEG_X, (10, iy * 20, 10)),
                PortPolarity.ANTI_STUD,
                ConnectorType.TECHNIC_PIN_HOLE,
            )
        )
        pid += 1
        ports.append(
            PortSpec(
                pid,
                LDUFrame(ROT_Z_TO_POS_X, (-10, iy * 20, 10)),
                PortPolarity.ANTI_STUD,
                ConnectorType.TECHNIC_PIN_HOLE,
            )
        )
        pid += 1
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(b_left, b_right),
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=True,
    )


def _make_technic_axle(
    part_id: str,
    name: str,
    length_studs: int,
    mass_grams: float,
) -> BrickPrimitive:
    """Technic Axle (`32062` 2L, `4519` 3L, `3705` 4L, `32073` 5L, `3706` 6L, `3707` 8L)."""
    len_ldu = length_studs * 20
    shaft = LDUBox(-4, -4, 0, 4, 4, len_ldu)
    ports: List[PortSpec] = []
    pid = 0
    for k in range(length_studs):
        zc = 10 + 20 * k
        ports.append(
            PortSpec(pid, LDUFrame(0, (0, 0, zc)), PortPolarity.STUD, ConnectorType.TECHNIC_AXLE)
        )
        pid += 1
        ports.append(
            PortSpec(
                pid,
                LDUFrame(ROT_Z_INVERT, (0, 0, zc)),
                PortPolarity.STUD,
                ConnectorType.TECHNIC_AXLE,
            )
        )
        pid += 1
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(shaft,),
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=True,
    )


def _make_window_or_door_frame(
    part_id: str,
    name: str,
    studs_x: int,
    studs_y: int,
    bricks_z: int,
    mass_grams: float,
    top_studs: bool = True,
) -> BrickPrimitive:
    """Hollow architectural window/door frame leaving interior pane opening unobstructed."""
    h_ldu = bricks_z * BRICK_HEIGHT_LDU
    xmax = (studs_x - 1) * 20 + 10
    ymax = (studs_y - 1) * 20 + 10
    sill = LDUBox(-10, -10, 0, xmax, ymax, 4)
    header = LDUBox(-10, -10, h_ldu - 4, xmax, ymax, h_ldu)
    jamb_l = LDUBox(-10, -10, 4, xmax, -6, h_ldu - 4)
    jamb_r = LDUBox(-10, ymax - 4, 4, xmax, ymax, h_ldu - 4)
    ports: List[PortSpec] = []
    pid = 0
    for ix in range(studs_x):
        for iy in range(studs_y):
            ports.append(PortSpec(pid, LDUFrame(0, (ix * 20, iy * 20, 0)), PortPolarity.ANTI_STUD))
            pid += 1
    if top_studs:
        for ix in range(studs_x):
            for iy in range(studs_y):
                ports.append(
                    PortSpec(
                        pid,
                        LDUFrame(0, (ix * 20, iy * 20, h_ldu)),
                        PortPolarity.STUD,
                        ConnectorType.HOLLOW_STUD,
                    )
                )
                pid += 1
    # Interior snap clip at mid-height for pane/door insert
    ports.append(
        PortSpec(
            pid,
            LDUFrame(ROT_Z_TO_NEG_X, (0, (studs_y - 1) * 10, h_ldu // 2)),
            PortPolarity.ANTI_STUD,
            ConnectorType.CLIP_3_18MM,
        )
    )
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(sill, header, jamb_l, jamb_r),
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=True,
    )


def _make_pane_or_door_insert(
    part_id: str,
    name: str,
    studs_y: int,
    bricks_z: int,
    mass_grams: float,
) -> BrickPrimitive:
    """Thin glass pane or door leaf snapping inside a window/door frame."""
    h_ldu = bricks_z * BRICK_HEIGHT_LDU
    ymax = (studs_y - 1) * 20 + 6
    pane_box = LDUBox(-2, -6, 4, 2, ymax, h_ldu - 4)
    ports = (
        PortSpec(
            0,
            LDUFrame(ROT_Z_TO_POS_X, (0, (studs_y - 1) * 10, h_ldu // 2)),
            PortPolarity.STUD,
            ConnectorType.BAR_3_18MM,
        ),
    )
    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(pane_box,),
        ports=ports,
        mass_grams=mass_grams,
        is_snot=True,
    )


def build_tier_a_catalog(seed_catalog: Dict[str, BrickPrimitive]) -> Dict[str, BrickPrimitive]:
    """Build the 160+ part `TIER_A_CATALOG` covering 100% of Set 10698-1 + OMR + 8 connector families."""
    cat: Dict[str, BrickPrimitive] = dict(seed_catalog)

    def _reg(p: BrickPrimitive) -> None:
        cat[p.part_id] = p

    # --- 1. Additional Bricks (Set 10698-1 + OMR) ---
    _reg(_make_rect_part("3065", "Brick 1x2 without Bottom Tube", 1, 2, 24, 0.78))
    _reg(_make_rect_part("6223", "Brick 2x2 without Inside Ridges", 2, 2, 24, 1.48))
    _reg(_make_rect_part("3245c", "Brick 1x2x2 with Inside Stud Holder", 1, 2, 48, 1.56))
    _reg(_make_rect_part("3006", "Brick 2x10", 2, 10, 24, 7.20))
    _reg(_make_rect_part("30136", "Brick Special 1x2 Palisade", 1, 2, 24, 0.80))
    _reg(_make_rect_part("2877", "Brick Special 1x2 with Grille", 1, 2, 24, 0.80))
    # 2x2 Corner Brick (2357)
    _reg(
        BrickPrimitive(
            part_id="2357",
            name="Brick 2x2 Corner",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 24), LDUBox(10, -10, 0, 30, 10, 24)),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(0, (0, 20, 24)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (20, 0, 24)), PortPolarity.STUD),
            ),
            mass_grams=1.15,
        )
    )

    # --- 2. Additional Plates (Set 10698-1 + OMR) ---
    _reg(_make_rect_part("3666", "Plate 1x6", 1, 6, 8, 1.02))
    _reg(_make_rect_part("3460", "Plate 1x8", 1, 8, 8, 1.35))
    _reg(_make_rect_part("4477", "Plate 1x10", 1, 10, 8, 1.68))
    _reg(_make_rect_part("60479", "Plate 1x12", 1, 12, 8, 2.02))
    _reg(_make_rect_part("3795", "Plate 2x6", 2, 6, 8, 1.95))
    _reg(_make_rect_part("3034", "Plate 2x8", 2, 8, 8, 2.60))
    _reg(_make_rect_part("3832", "Plate 2x10", 2, 10, 8, 3.25))
    _reg(_make_rect_part("11212", "Plate 3x3", 3, 3, 8, 1.48))
    _reg(_make_rect_part("3031", "Plate 4x4", 4, 4, 8, 2.62))
    _reg(_make_rect_part("3032", "Plate 4x6", 4, 6, 8, 3.90))
    _reg(_make_rect_part("3029", "Plate 4x12", 4, 12, 8, 7.80))
    _reg(_make_rect_part("3028", "Plate 6x12", 6, 12, 8, 11.70))
    _reg(_make_rect_part("91405", "Plate 16x16", 16, 16, 8, 41.50))
    _reg(_make_rect_part("4032a", "Plate Round 2x2 with Axle Hole", 2, 2, 8, 0.58))
    # 15397 Plate 3x3 Cross (5 studs)
    _reg(
        BrickPrimitive(
            part_id="15397",
            name="Plate 3x3 Cross",
            hull_boxes=(LDUBox(10, -10, 0, 30, 50, 8), LDUBox(-10, 10, 0, 50, 30, 8)),
            ports=(
                PortSpec(0, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (20, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 40, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(0, (40, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(5, LDUFrame(0, (20, 0, 8)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(0, (20, 20, 8)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(0, (20, 40, 8)), PortPolarity.STUD),
                PortSpec(8, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(9, LDUFrame(0, (40, 20, 8)), PortPolarity.STUD),
            ),
            mass_grams=0.85,
        )
    )
    # 2420 Plate 2x2 Corner
    _reg(
        BrickPrimitive(
            part_id="2420",
            name="Plate 2x2 Corner",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8), LDUBox(10, -10, 0, 30, 10, 8)),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (20, 0, 8)), PortPolarity.STUD),
            ),
            mass_grams=0.49,
        )
    )
    # Jumper Plates 1x2 with 1 Center Stud (3794a, 15573)
    for jpid, jname in (("3794a", "Plate 1x2 with 1 Stud"), ("15573", "Plate 1x2 Jumper")):
        _reg(
            BrickPrimitive(
                part_id=jpid,
                name=jname,
                hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8),),
                ports=(
                    PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                    PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                    PortSpec(2, LDUFrame(0, (0, 10, 8)), PortPolarity.STUD, ConnectorType.HOLLOW_STUD),
                ),
                mass_grams=0.33,
                is_snot=True,
            )
        )

    # --- 3. Tiles (Set 10698-1 + OMR) ---
    _reg(_make_tile_part("3070b", "Tile 1x1 with Groove", 1, 1, 0.15))
    _reg(_make_tile_part("3069b", "Tile 1x2 with Groove", 1, 2, 0.30))
    _reg(_make_tile_part("2412b", "Tile Special 1x2 Grille", 1, 2, 0.28))
    _reg(_make_tile_part("63864", "Tile 1x3", 1, 3, 0.45))
    _reg(_make_tile_part("2431", "Tile 1x4 with Groove", 1, 4, 0.60))
    _reg(_make_tile_part("6636", "Tile 1x6", 1, 6, 0.90))
    _reg(_make_tile_part("4162", "Tile 1x8", 1, 8, 1.20))
    _reg(_make_tile_part("3068b", "Tile 2x2 with Groove", 2, 2, 0.58))
    _reg(_make_tile_part("87079", "Tile 2x4 with Groove", 2, 4, 1.16))
    _reg(_make_tile_part("14769", "Tile Round 2x2", 2, 2, 0.52))
    _reg(_make_tile_part("98138pr0008", "Tile Round 1x1 Eye Open", 1, 1, 0.14))
    _reg(_make_tile_part("98138pr0026", "Tile Round 1x1 Eye Closed", 1, 1, 0.14))

    # --- 4. Slopes, Curves & Arches (Set 10698-1 + OMR) ---
    _reg(_make_slope_part("3040b", "Slope 45 2x1", 2, 1, 24, 1, 0.62))
    _reg(_make_slope_part("60481", "Slope 65 2x1x2", 2, 1, 48, 1, 1.15))
    _reg(_make_slope_part("3037", "Slope 45 2x4", 2, 4, 24, 1, 2.35))
    _reg(_make_slope_part("3045", "Slope 45 2x2 Double Convex", 2, 2, 24, 1, 1.05))
    _reg(_make_slope_part("4286", "Slope 33 3x1", 3, 1, 24, 1, 0.88))
    _reg(_make_slope_part("54200", "Slope 30 1x1x2/3 Cheese", 1, 1, 16, 0, 0.25))
    _reg(_make_slope_part("85984", "Slope 30 1x2x2/3", 1, 2, 16, 0, 0.50))
    _reg(_make_slope_part("11477", "Slope Curved 2x1 No Studs", 2, 1, 16, 0, 0.42))
    _reg(_make_slope_part("15068", "Slope Curved 2x2 No Studs", 2, 2, 16, 0, 0.84))
    _reg(_make_slope_part("50950", "Brick Curved 3x1 No Studs", 3, 1, 24, 0, 0.92))
    _reg(_make_slope_part("6091", "Brick Curved 1x2x1 1/3", 2, 1, 32, 1, 0.78))
    _reg(_make_slope_part("93273", "Brick Curved 4x1 Double No Studs", 4, 1, 16, 0, 1.08))
    _reg(_make_slope_part("6081", "Brick Curved 2x4x1 1/3", 4, 2, 32, 1, 2.65))
    _reg(_make_slope_part("6215", "Brick Curved 2x3", 3, 2, 24, 1, 1.85))
    _reg(_make_slope_part("30165", "Brick Curved 2x2 Two Top Studs", 2, 2, 24, 1, 1.30))
    _reg(_make_slope_part("24309", "Slope Curved 3x2 No Studs", 3, 2, 16, 0, 1.24))
    _reg(_make_inverted_slope_part("3660", "Slope Inverted 45 2x2", 2, 1, 2, 24, 1.16))
    _reg(_make_inverted_slope_part("76959", "Slope Inverted 45 2x2 Hollow Pin", 2, 1, 2, 24, 1.16))
    _reg(_make_inverted_slope_part("3665", "Slope Inverted 45 2x1", 2, 1, 1, 24, 0.60))
    _reg(_make_inverted_slope_part("3676", "Slope Inverted 45 2x2 Double Convex", 2, 1, 2, 24, 1.12))
    _reg(_make_inverted_slope_part("4287c", "Slope Inverted 33 3x1", 3, 1, 1, 24, 0.84))
    _reg(_make_arch_part("3659", "Brick Arch 1x4", 4, 24, 1.28))
    _reg(_make_arch_part("15254", "Brick Arch 1x6x2", 6, 48, 2.75))

    # --- 5. Round Bricks, Cones, Flowers & Plants (Set 10698-1 + OMR) ---
    for rpid, rname, h, m in (
        ("6141", "Plate Round 1x1", 8, 0.16),
        ("3062b", "Brick Round 1x1 Open Stud", 24, 0.40),
        ("59900", "Cone 1x1 Top Groove", 24, 0.34),
        ("33291", "Flower Plate Round 1x1 4 Petals", 8, 0.17),
    ):
        _reg(
            BrickPrimitive(
                part_id=rpid,
                name=rname,
                hull_boxes=(LDUBox(-10, -10, 0, 10, 10, h),),
                ports=(
                    PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD, ConnectorType.HOLLOW_STUD),
                    PortSpec(1, LDUFrame(0, (0, 0, h)), PortPolarity.STUD, ConnectorType.HOLLOW_STUD),
                ),
                mass_grams=m,
            )
        )
    _reg(_make_rect_part("85080", "Brick Round Corner 2x2 Macaroni", 2, 2, 24, 1.18))
    _reg(_make_slope_part("30367c", "Brick Round 2x2 Dome Top", 2, 2, 24, 1, 1.10))
    _reg(_make_slope_part("30151b", "Brick Round 2x2x1 2/3 Dome Top", 2, 2, 40, 1, 1.65))
    _reg(_make_slope_part("3942c", "Cone 2x2x2 Open Stud", 2, 2, 48, 1, 1.55))
    _reg(_make_rect_part("87081", "Brick Round 4x4 Centre Hole", 4, 4, 24, 4.10))
    _reg(_make_rect_part("4727", "Plant Flower 2x2 Leaves", 2, 2, 8, 0.45))
    _reg(_make_rect_part("98262", "Plant Flower 2x2 Round", 2, 2, 8, 0.44))

    # --- 6. SNOT Bricks & Brackets (Set 10698-1 + OMR) ---
    # 47905 Brick 1x1 with Studs on 2 Opposite Sides (+X and -X)
    _reg(
        BrickPrimitive(
            part_id="47905",
            name="Brick 1x1 with Studs on 2 Sides",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 10, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(2, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 16)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(ROT_Z_TO_NEG_X, (-10, 0, 16)), PortPolarity.STUD),
            ),
            mass_grams=0.49,
            is_snot=True,
        )
    )
    # 30414 Brick 1x4 with 4 Studs on One Side (+X)
    _reg(
        BrickPrimitive(
            part_id="30414",
            name="Brick 1x4 with 4 Studs on 1 Side",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 70, 24),),
            ports=tuple(
                [PortSpec(iy, LDUFrame(0, (0, iy * 20, 0)), PortPolarity.ANTI_STUD) for iy in range(4)]
                + [PortSpec(4 + iy, LDUFrame(0, (0, iy * 20, 24)), PortPolarity.STUD) for iy in range(4)]
                + [
                    PortSpec(8 + iy, LDUFrame(ROT_Z_TO_POS_X, (10, iy * 20, 16)), PortPolarity.STUD)
                    for iy in range(4)
                ]
            ),
            mass_grams=1.68,
            is_snot=True,
        )
    )
    # 44728 Bracket 1x2 - 2x2 Upward
    _reg(
        BrickPrimitive(
            part_id="44728",
            name="Bracket 1x2 - 2x2",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8), LDUBox(10, -10, 0, 14, 30, 40)),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(ROT_Z_TO_POS_X, (14, 0, 12)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(ROT_Z_TO_POS_X, (14, 20, 12)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(ROT_Z_TO_POS_X, (14, 0, 32)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(ROT_Z_TO_POS_X, (14, 20, 32)), PortPolarity.STUD),
            ),
            mass_grams=0.86,
            is_snot=True,
        )
    )
    # 99206 Plate Special 2x2x2/3 with Two Studs on Side
    _reg(
        BrickPrimitive(
            part_id="99206",
            name="Plate Special 2x2x2/3 with 2 Side Studs",
            hull_boxes=(LDUBox(-10, -10, 0, 30, 30, 16),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (20, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(0, (0, 0, 16)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (0, 20, 16)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(ROT_Z_TO_POS_X, (30, 0, 8)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(ROT_Z_TO_POS_X, (30, 20, 8)), PortPolarity.STUD),
            ),
            mass_grams=0.92,
            is_snot=True,
        )
    )
    # 22885 Brick Special 1x2x1 2/3 with 8 Side Studs
    _reg(
        BrickPrimitive(
            part_id="22885",
            name="Brick Special 1x2x1 2/3 with Studs on 2 Sides",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 40),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 40)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 40)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 12)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(ROT_Z_TO_POS_X, (10, 20, 12)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 32)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(ROT_Z_TO_POS_X, (10, 20, 32)), PortPolarity.STUD),
            ),
            mass_grams=1.18,
            is_snot=True,
        )
    )

    # --- 7. Clips, Bars, Fences, Panels & Windows/Doors (Set 10698-1 + OMR) ---
    for cpid, cname in (
        ("4081b", "Plate 1x1 with Clip Light Ring"),
        ("61252", "Plate 1x1 with Horizontal Clip"),
        ("6019", "Plate 1x1 with Horizontal U-Clip"),
        ("2555", "Tile 1x1 with Clip"),
    ):
        top_ports = (
            ()
            if cpid == "2555"
            else (PortSpec(1, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),)
        )
        _reg(
            BrickPrimitive(
                part_id=cpid,
                name=cname,
                hull_boxes=(LDUBox(-10, -10, 0, 10, 10, 8), LDUBox(10, -4, 0, 12, 4, 8)),
                ports=(
                    (PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),)
                    + top_ports
                    + (
                        PortSpec(
                            len(top_ports) + 1,
                            LDUFrame(ROT_Z_TO_NEG_X, (16, 0, 4)),
                            PortPolarity.ANTI_STUD,
                            ConnectorType.CLIP_3_18MM,
                        ),
                    )
                ),
                mass_grams=0.23,
                is_snot=True,
            )
        )
    # 3957b Antenna 1x4 (1x1 base anti-stud + vertical 3.18mm bar shaft)
    _reg(
        BrickPrimitive(
            part_id="3957b",
            name="Antenna 1x4 with Flat Top",
            hull_boxes=(LDUBox(-8, -8, 0, 8, 8, 8), LDUBox(-3, -3, 8, 3, 3, 96)),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(ROT_Z_TO_POS_X, (0, 0, 48)), PortPolarity.STUD, ConnectorType.BAR_3_18MM),
                PortSpec(2, LDUFrame(0, (0, 0, 96)), PortPolarity.STUD, ConnectorType.BAR_3_18MM),
            ),
            mass_grams=0.40,
            is_snot=True,
        )
    )
    _reg(_make_rect_part("15332", "Fence Spindled 1x4x2", 1, 4, 48, 1.95))
    _reg(_make_rect_part("15207", "Panel 1x4x1", 1, 4, 24, 0.98))
    _reg(_make_rect_part("87544", "Panel 1x2x3", 1, 2, 72, 1.85))
    _reg(_make_window_or_door_frame("60592", "Window 1x2x2 Flat Front", 1, 2, 2, 1.10))
    _reg(_make_pane_or_door_insert("60601", "Glass for Window 1x2x2", 2, 2, 0.42))
    _reg(_make_window_or_door_frame("60598", "Window 2x4x3 Frame", 2, 4, 3, 2.85))
    _reg(_make_pane_or_door_insert("60607", "Window 1x2x3 Pane Latticed", 2, 3, 0.65))
    _reg(_make_pane_or_door_insert("60608", "Window 1x2x3 Pane", 2, 3, 0.65))
    _reg(_make_window_or_door_frame("60593", "Window 1x2x3 Flat Front", 1, 2, 3, 1.48))
    _reg(_make_pane_or_door_insert("60602", "Glass for Window 1x2x3", 2, 3, 0.62))
    _reg(_make_window_or_door_frame("30044", "Window 1x2x2 2/3 Rounded Top", 1, 2, 2, 1.25))
    _reg(_make_pane_or_door_insert("30046", "Window 1x2x2 2/3 Lattice Pane", 2, 2, 0.48))
    _reg(_make_window_or_door_frame("90195", "Window 1x2x2 Castle", 1, 2, 2, 1.15))
    _reg(_make_window_or_door_frame("60596", "Door Frame 1x4x6", 1, 4, 6, 4.20))
    _reg(_make_window_or_door_frame("60599", "Door Frame 2x4x6", 2, 4, 6, 5.40))
    _reg(_make_pane_or_door_insert("60621", "Door 1x4x6 Barred", 4, 6, 2.60))
    _reg(_make_pane_or_door_insert("60623", "Door 1x4x6 4 Panes", 4, 6, 2.80))

    # --- 8. Technic Bricks, Liftarms, Pins, Axles & Wheels (Set 10698-1 + OMR) ---
    # 3941 Brick Round 2x2 with Vertical Axle Hole
    _reg(
        BrickPrimitive(
            part_id="3941",
            name="Brick Round 2x2 with Axle Hole",
            hull_boxes=(
                LDUBox(-10, -10, 0, 30, 4, 24),
                LDUBox(-10, 16, 0, 30, 30, 24),
                LDUBox(-10, 4, 0, 4, 16, 24),
                LDUBox(16, 4, 0, 30, 16, 24),
            ),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (20, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (0, 20, 24)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(0, (20, 0, 24)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(0, (20, 20, 24)), PortPolarity.STUD),
                PortSpec(
                    8,
                    LDUFrame(0, (10, 10, 10)),
                    PortPolarity.ANTI_STUD,
                    ConnectorType.TECHNIC_AXLE_HOLE,
                ),
            ),
            mass_grams=1.32,
            is_snot=True,
        )
    )
    # 2458 Brick Special 1x2 with Technic Pin (+X)
    _reg(
        BrickPrimitive(
            part_id="2458",
            name="Brick Special 1x2 with Pin",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 24)), PortPolarity.STUD),
                PortSpec(
                    4,
                    LDUFrame(ROT_Z_TO_POS_X, (10, 10, 14)),
                    PortPolarity.STUD,
                    ConnectorType.SMOOTH_PIN,
                ),
            ),
            mass_grams=0.95,
            is_snot=True,
        )
    )
    # 6232 Brick Special 2x2 with Pin and Axle Hole
    _reg(
        BrickPrimitive(
            part_id="6232",
            name="Brick Special 2x2 with Pin and Axle Hole",
            hull_boxes=(
                LDUBox(-10, -10, 0, 30, 4, 24),
                LDUBox(-10, 16, 0, 30, 30, 24),
                LDUBox(-10, 4, 0, 4, 16, 24),
                LDUBox(16, 4, 0, 30, 16, 24),
            ),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (20, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (0, 20, 24)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(0, (20, 0, 24)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(0, (20, 20, 24)), PortPolarity.STUD),
                PortSpec(
                    8,
                    LDUFrame(ROT_Z_TO_POS_X, (30, 10, 14)),
                    PortPolarity.STUD,
                    ConnectorType.SMOOTH_PIN,
                ),
                PortSpec(
                    9,
                    LDUFrame(0, (10, 10, 10)),
                    PortPolarity.ANTI_STUD,
                    ConnectorType.TECHNIC_AXLE_HOLE,
                ),
            ),
            mass_grams=1.64,
            is_snot=True,
        )
    )
    _reg(_make_technic_brick("3700", "Technic Brick 1x2 with Hole", 2, 0.82))
    _reg(_make_technic_brick("32064a", "Technic Brick 1x2 with Axle Hole", 2, 0.84))
    _reg(_make_technic_brick("3701", "Technic Brick 1x4 with Holes", 4, 1.58))
    _reg(_make_technic_brick("3894", "Technic Brick 1x6 with Holes", 6, 2.35))
    _reg(_make_technic_brick("3702", "Technic Brick 1x8 with Holes", 8, 3.12))
    _reg(_make_technic_brick("3703", "Technic Brick 1x16 with Holes", 16, 6.20))
    _reg(_make_technic_liftarm("32523", "Technic Liftarm 1x3 Thick", 3, 0.85))
    _reg(_make_technic_liftarm("32316", "Technic Liftarm 1x5 Thick", 5, 1.40))
    _reg(_make_technic_liftarm("32524", "Technic Liftarm 1x7 Thick", 7, 1.96))
    _reg(_make_technic_axle("32062", "Technic Axle 2L Notched", 2, 0.24))
    _reg(_make_technic_axle("4519", "Technic Axle 3L", 3, 0.36))
    _reg(_make_technic_axle("3705", "Technic Axle 4L", 4, 0.48))
    _reg(_make_technic_axle("32073", "Technic Axle 5L", 5, 0.60))
    _reg(_make_technic_axle("3706", "Technic Axle 6L", 6, 0.72))
    _reg(_make_technic_axle("3707", "Technic Axle 8L", 8, 0.96))
    # Additional Technic Pins / Bushes
    _reg(
        BrickPrimitive(
            part_id="4459",
            name="Technic Pin with Friction Ridges",
            hull_boxes=(LDUBox(-4, -4, -4, 4, 4, 4),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
                PortSpec(1, LDUFrame(ROT_Z_INVERT, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
            ),
            mass_grams=0.16,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="6558",
            name="Technic Pin 3L with Friction Ridges",
            hull_boxes=(LDUBox(-4, -4, -4, 4, 4, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
                PortSpec(1, LDUFrame(ROT_Z_INVERT, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
                PortSpec(2, LDUFrame(0, (0, 0, 20)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
            ),
            mass_grams=0.28,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="43093",
            name="Technic Axle 1L with Pin with Friction Ridges",
            hull_boxes=(LDUBox(-4, -4, -4, 4, 4, 14),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
                PortSpec(1, LDUFrame(ROT_Z_INVERT, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_AXLE),
            ),
            mass_grams=0.20,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="4274",
            name="Technic Pin 1/2 with Stud",
            hull_boxes=(LDUBox(-4, -4, -4, 4, 4, 4),),
            ports=(
                PortSpec(0, LDUFrame(ROT_Z_INVERT, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
                PortSpec(1, LDUFrame(0, (0, 0, 4)), PortPolarity.STUD, ConnectorType.HOLLOW_STUD),
            ),
            mass_grams=0.14,
            is_snot=True,
        )
    )
    for bpid, bname, bh in (("3713", "Technic Bush", 20), ("32123a", "Technic Bush 1/2", 10), ("6536", "Technic Axle and Pin Connector", 20)):
        _reg(
            BrickPrimitive(
                part_id=bpid,
                name=bname,
                hull_boxes=(LDUBox(-8, -8, 0, 8, -4, bh), LDUBox(-8, 4, 0, 8, 8, bh)),
                ports=(
                    PortSpec(0, LDUFrame(0, (0, 0, bh // 2)), PortPolarity.ANTI_STUD, ConnectorType.TECHNIC_AXLE_HOLE),
                    PortSpec(1, LDUFrame(ROT_Z_INVERT, (0, 0, bh // 2)), PortPolarity.ANTI_STUD, ConnectorType.TECHNIC_AXLE_HOLE),
                ),
                mass_grams=0.22,
                is_snot=True,
            )
        )

    # Wheels & Tyres (Set 10698-1: 2926 holder, 6014b & 55981 wheels, 87697 & 58090 tyres)
    _reg(
        BrickPrimitive(
            part_id="2926",
            name="Plate Special 1x4 with Wheels Holder",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 70, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 40, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (0, 60, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(0, (0, 40, 8)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(0, (0, 60, 8)), PortPolarity.STUD),
                PortSpec(8, LDUFrame(ROT_Z_TO_POS_X, (14, 30, 4)), PortPolarity.STUD, ConnectorType.WHEEL_PIN),
                PortSpec(9, LDUFrame(ROT_Z_TO_NEG_X, (-14, 30, 4)), PortPolarity.STUD, ConnectorType.WHEEL_PIN),
            ),
            mass_grams=0.88,
            is_snot=True,
        )
    )
    for wpid, wname, r_ldu, w_ldu, wm in (
        ("6014b", "Wheel 11x12", 12, 24, 0.90),
        ("55981", "Wheel 18x14 with Pin Hole", 18, 28, 1.75),
    ):
        _reg(
            BrickPrimitive(
                part_id=wpid,
                name=wname,
                hull_boxes=(LDUBox(2, -r_ldu, -r_ldu, 2 + w_ldu, r_ldu, r_ldu),),
                ports=(
                    PortSpec(0, LDUFrame(ROT_Z_TO_NEG_X, (0, 0, 0)), PortPolarity.ANTI_STUD, ConnectorType.WHEEL_HUB_HOLE),
                    PortSpec(1, LDUFrame(ROT_Z_TO_POS_X, (2 + w_ldu, 0, 0)), PortPolarity.STUD, ConnectorType.WHEEL_PIN),
                ),
                mass_grams=wm,
                is_snot=True,
            )
        )
    for tpid, tname, r_out, r_in, w_ldu, tm in (
        ("87697", "Tyre 21x12", 24, 14, 24, 2.20),
        ("58090", "Tyre 30.4x14", 36, 20, 28, 4.80),
    ):
        _reg(
            BrickPrimitive(
                part_id=tpid,
                name=tname,
                hull_boxes=(
                    LDUBox(0, -r_out, r_in, w_ldu, r_out, r_out),
                    LDUBox(0, -r_out, -r_out, w_ldu, r_out, -r_in),
                ),
                ports=(
                    PortSpec(0, LDUFrame(ROT_Z_TO_NEG_X, (w_ldu, 0, 0)), PortPolarity.ANTI_STUD, ConnectorType.WHEEL_HUB_HOLE),
                ),
                mass_grams=tm,
                is_snot=True,
            )
        )

    # --- 9. Hinges, Turntables, Ball Joints & Brick Separator (Set 10698-1 + OMR) ---
    # 3680 Turntable 2x2 Base + 3679 Turntable 2x2 Top
    _reg(
        BrickPrimitive(
            part_id="3680",
            name="Turntable 2x2 Plate Base",
            hull_boxes=(LDUBox(-10, -10, 0, 30, 30, 4),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (20, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(ROT_Z_INVERT, (10, 10, 4)), PortPolarity.ANTI_STUD, ConnectorType.TURNTABLE_BASE_SOCKET),
            ),
            mass_grams=0.58,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="3679",
            name="Turntable 2x2 Plate Top",
            hull_boxes=(LDUBox(-10, -10, 4, 30, 30, 8),),
            ports=(
                PortSpec(0, LDUFrame(ROT_Z_INVERT, (10, 10, 4)), PortPolarity.STUD, ConnectorType.TURNTABLE_TOP_PIN),
                PortSpec(1, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(2, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (20, 0, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(0, (20, 20, 8)), PortPolarity.STUD),
            ),
            mass_grams=0.56,
            is_snot=True,
        )
    )

    # Set 10698-1 9-Tooth Locking Hinges: 60471 (2-finger side socket), 92582 (1-finger top pin), 30387 (1x4 hinge brick)
    _reg(
        BrickPrimitive(
            part_id="60471",
            name="Hinge Plate 1x2 Locking with 2 Fingers on Side",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(
                    4,
                    LDUFrame(ROT_Z_TO_POS_Y, (16, 10, 4)),
                    PortPolarity.ANTI_STUD,
                    ConnectorType.LOCKING_HINGE_SOCKET,
                ),
            ),
            mass_grams=0.45,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="92582",
            name="Hinge Plate 2x2 Locking with 1 Finger on Top",
            hull_boxes=(LDUBox(-10, -10, 0, 30, 30, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (20, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(0, (20, 0, 8)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (20, 20, 8)), PortPolarity.STUD),
                PortSpec(
                    6,
                    LDUFrame(ROT_Z_TO_POS_Y, (0, 10, 14)),
                    PortPolarity.STUD,
                    ConnectorType.LOCKING_HINGE_PIN,
                ),
            ),
            mass_grams=0.72,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="30387",
            name="Hinge Brick 1x4 Locking",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 70, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 40, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (0, 60, 0)), PortPolarity.ANTI_STUD),
                PortSpec(4, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(0, (0, 20, 24)), PortPolarity.STUD),
                PortSpec(6, LDUFrame(0, (0, 40, 24)), PortPolarity.STUD),
                PortSpec(7, LDUFrame(0, (0, 60, 24)), PortPolarity.STUD),
                PortSpec(8, LDUFrame(ROT_Z_TO_POS_X, (14, -10, 12)), PortPolarity.STUD, ConnectorType.LOCKING_HINGE_PIN),
                PortSpec(9, LDUFrame(ROT_Z_TO_POS_X, (14, 70, 12)), PortPolarity.ANTI_STUD, ConnectorType.LOCKING_HINGE_SOCKET),
            ),
            mass_grams=1.74,
            is_snot=True,
        )
    )

    # Free / Smooth Hinges (3937 Base + 3938 Top, 2429 + 2430)
    _reg(
        BrickPrimitive(
            part_id="3937",
            name="Hinge Brick 1x2 Base (Smooth)",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 16),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(
                    2,
                    LDUFrame(ROT_Z_TO_POS_Y, (0, 10, 20)),
                    PortPolarity.ANTI_STUD,
                    ConnectorType.HINGE_SOCKET,
                ),
            ),
            mass_grams=0.65,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="3938",
            name="Hinge Brick 1x2 Top Plate (Smooth)",
            hull_boxes=(LDUBox(-10, -10, 22, 10, 30, 24),),
            ports=(
                PortSpec(
                    0,
                    LDUFrame(ROT_Z_TO_POS_Y, (0, 10, 20)),
                    PortPolarity.STUD,
                    ConnectorType.HINGE_PIN,
                ),
                PortSpec(1, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(2, LDUFrame(0, (0, 20, 24)), PortPolarity.STUD),
            ),
            mass_grams=0.38,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="2429",
            name="Hinge Plate 1x4 Swivel Base",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 25, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(0, (0, 30, 4)), PortPolarity.ANTI_STUD, ConnectorType.HINGE_SOCKET),
            ),
            mass_grams=0.40,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="2430",
            name="Hinge Plate 1x4 Swivel Top",
            hull_boxes=(LDUBox(-10, 35, 0, 10, 70, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 30, 4)), PortPolarity.STUD, ConnectorType.HINGE_PIN),
                PortSpec(1, LDUFrame(0, (0, 40, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 60, 0)), PortPolarity.ANTI_STUD),
                PortSpec(3, LDUFrame(0, (0, 40, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(0, (0, 60, 8)), PortPolarity.STUD),
            ),
            mass_grams=0.40,
            is_snot=True,
        )
    )

    # Mixel Towball Joints & Sockets (14417 Ball Plate, 14418 Socket Plate, 14704 Mid Socket)
    _reg(
        BrickPrimitive(
            part_id="14417",
            name="Plate Special 1x2 with Towball on End",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(ROT_Z_TO_POS_X, (22, 10, 4)), PortPolarity.STUD, ConnectorType.BALL_JOINT),
            ),
            mass_grams=0.48,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="14418",
            name="Plate Special 1x2 with Towball Socket on End",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(ROT_Z_TO_NEG_X, (22, 10, 4)), PortPolarity.ANTI_STUD, ConnectorType.BALL_SOCKET),
            ),
            mass_grams=0.54,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="14704",
            name="Plate Special 1x2 with Center Towball Socket",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(ROT_Z_TO_NEG_X, (18, 10, 4)), PortPolarity.ANTI_STUD, ConnectorType.BALL_SOCKET),
            ),
            mass_grams=0.52,
            is_snot=True,
        )
    )

    # 96874 Brick and Axle Separator v2.0 (2x1 anti-stud head + top axle extractor pin)
    _reg(
        BrickPrimitive(
            part_id="96874",
            name="Brick and Axle Separator v2.0",
            hull_boxes=(LDUBox(-10, -10, 0, 30, 90, 8),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (20, 0, 8)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(0, (10, 30, 8)), PortPolarity.STUD, ConnectorType.TECHNIC_AXLE),
            ),
            mass_grams=3.10,
            is_snot=True,
        )
    )

    # --- 10. Additional High-Frequency OMR System / SNOT / Technic Parts ---
    _reg(_make_rect_part("6111", "Brick 1x10", 1, 10, 24, 3.88))
    _reg(_make_rect_part("30357", "Plate Round Corner 3x3", 3, 3, 8, 1.32))
    _reg(_make_rect_part("41767", "Wedge Plate 4x2 Right", 2, 4, 8, 0.95))
    _reg(_make_rect_part("41768", "Wedge Plate 4x2 Left", 2, 4, 8, 0.95))
    _reg(_make_rect_part("11215", "Bracket 5x2x1 1/3", 2, 5, 16, 1.40))
    _reg(_make_slope_part("6191", "Brick Curved 1x4x1 1/3", 4, 1, 32, 0, 1.45))
    _reg(_make_slope_part("4445", "Slope 45 2x8", 2, 8, 24, 1, 4.70))
    _reg(_make_technic_axle("3708", "Technic Axle 12L", 12, 1.44))
    _reg(_make_technic_liftarm("32140", "Technic Liftarm 2x4 L-Shape", 4, 1.35))
    _reg(_make_technic_brick("32530", "Technic Pin Connector Plate 1x2", 2, 0.76))
    _reg(_make_rect_part("3035", "Plate 4x8", 4, 8, 8, 5.20))
    _reg(_make_rect_part("3958", "Plate 6x6", 6, 6, 8, 5.85))
    _reg(_make_rect_part("3036", "Plate 6x8", 6, 8, 8, 7.80))
    _reg(_make_rect_part("3033", "Plate 6x10", 6, 10, 8, 9.75))
    _reg(_make_rect_part("2445", "Plate 2x12", 2, 12, 8, 3.90))
    _reg(_make_rect_part("30565", "Plate Round Corner 4x4", 4, 4, 8, 2.30))
    _reg(_make_rect_part("32028", "Plate 1x2 with Door Rail", 1, 2, 8, 0.38))
    _reg(_make_rect_part("4510", "Plate 1x8 with Door Rail", 1, 8, 8, 1.48))
    _reg(_make_rect_part("92593", "Plate Special 1x4 with 2 End Studs", 1, 4, 8, 0.65))
    _reg(_make_rect_part("87580", "Plate Special 2x2 with 1 Center Stud", 2, 2, 8, 0.64))
    _reg(_make_rect_part("4865b", "Panel 1x2x1", 1, 2, 24, 0.56))
    _reg(_make_rect_part("91501", "Panel 1x1x1 Corner", 1, 1, 24, 0.34))
    _reg(_make_tile_part("26603", "Tile 2x3", 2, 3, 0.88))
    _reg(_make_tile_part("3743", "Technic Gear Rack 1x4", 1, 4, 0.72))
    # 14719 Tile 2x2 Corner
    _reg(
        BrickPrimitive(
            part_id="14719",
            name="Tile 2x2 Corner",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8), LDUBox(10, -10, 0, 30, 10, 8)),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (20, 0, 0)), PortPolarity.ANTI_STUD),
            ),
            mass_grams=0.44,
        )
    )
    _reg(_make_slope_part("61678", "Slope Curved 4x1 No Studs", 4, 1, 24, 0, 1.15))
    _reg(_make_slope_part("3044b", "Slope 45 2x1 Double", 2, 1, 24, 0, 0.56))
    _reg(_make_arch_part("3308", "Brick Arch 1x8x2", 8, 48, 3.60))
    _reg(_make_technic_brick("3895", "Technic Brick 1x12 with Holes", 12, 4.65))
    _reg(_make_technic_axle("3737", "Technic Axle 10L", 10, 1.20))
    _reg(_make_rect_part("3738", "Technic Plate 2x8 with 7 Holes", 2, 8, 8, 2.45))
    for gpid, gname, gr in (("3647", "Technic Gear 8 Tooth", 10), ("4019", "Technic Gear 16 Tooth", 20)):
        _reg(
            BrickPrimitive(
                part_id=gpid,
                name=gname,
                hull_boxes=(LDUBox(-gr, -gr, 0, gr, -4, 8), LDUBox(-gr, 4, 0, gr, gr, 8)),
                ports=(
                    PortSpec(0, LDUFrame(0, (0, 0, 4)), PortPolarity.ANTI_STUD, ConnectorType.TECHNIC_AXLE_HOLE),
                    PortSpec(1, LDUFrame(ROT_Z_INVERT, (0, 0, 4)), PortPolarity.ANTI_STUD, ConnectorType.TECHNIC_AXLE_HOLE),
                ),
                mass_grams=0.45,
                is_snot=True,
            )
        )
    # 11211 Brick Special 1x2 with 2 Studs on 1 Side (+X)
    _reg(
        BrickPrimitive(
            part_id="11211",
            name="Brick Special 1x2 with 2 Studs on 1 Side",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
                PortSpec(2, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(0, (0, 20, 24)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 16)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(ROT_Z_TO_POS_X, (10, 20, 16)), PortPolarity.STUD),
            ),
            mass_grams=0.88,
            is_snot=True,
        )
    )
    # 32952 Brick Special 1x1x1 2/3 with 2 Studs on 1 Side (+X)
    _reg(
        BrickPrimitive(
            part_id="32952",
            name="Brick Special 1x1x1 2/3 with 2 Studs on 1 Side",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 10, 40),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 0, 40)), PortPolarity.STUD),
                PortSpec(2, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 12)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 32)), PortPolarity.STUD),
            ),
            mass_grams=0.68,
            is_snot=True,
        )
    )
    # 26604 Brick 1x1 with Studs on 2 Adjacent Sides (+X and +Y) & 4733 Travis Brick (4 Sides)
    _reg(
        BrickPrimitive(
            part_id="26604",
            name="Brick Special 1x1 with Studs on 2 Adjacent Sides",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 10, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(2, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 16)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(ROT_Z_TO_POS_Y, (0, 10, 16)), PortPolarity.STUD),
            ),
            mass_grams=0.48,
            is_snot=True,
        )
    )
    _reg(
        BrickPrimitive(
            part_id="4733",
            name="Brick Special 1x1 with Studs on 4 Sides",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 10, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(2, LDUFrame(ROT_Z_TO_POS_X, (10, 0, 16)), PortPolarity.STUD),
                PortSpec(3, LDUFrame(ROT_Z_TO_NEG_X, (-10, 0, 16)), PortPolarity.STUD),
                PortSpec(4, LDUFrame(ROT_Z_TO_POS_Y, (0, 10, 16)), PortPolarity.STUD),
                PortSpec(5, LDUFrame(ROT_Z_TO_NEG_Y, (0, -10, 16)), PortPolarity.STUD),
            ),
            mass_grams=0.52,
            is_snot=True,
        )
    )
    # 1x2 Plates/Tiles with Bar Handles or Clips (2432, 60478, 2540, 11476, 63868, 92280, 2921)
    for hpid, hname, top_s, ctype, pol in (
        ("2432", "Tile Special 1x2 with Handle", False, ConnectorType.BAR_3_18MM, PortPolarity.STUD),
        ("60478", "Plate Special 1x2 with Long Handle", True, ConnectorType.BAR_3_18MM, PortPolarity.STUD),
        ("2540", "Plate Special 1x2 with Handle", True, ConnectorType.BAR_3_18MM, PortPolarity.STUD),
        ("11476", "Plate Special 1x2 with Horizontal Clip", True, ConnectorType.CLIP_3_18MM, PortPolarity.ANTI_STUD),
        ("63868", "Plate Special 1x2 with End Clip", True, ConnectorType.CLIP_3_18MM, PortPolarity.ANTI_STUD),
        ("92280", "Plate Special 1x2 with Top Clip", True, ConnectorType.CLIP_3_18MM, PortPolarity.ANTI_STUD),
    ):
        p_list = [
            PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
            PortSpec(1, LDUFrame(0, (0, 20, 0)), PortPolarity.ANTI_STUD),
        ]
        if top_s:
            p_list.append(PortSpec(2, LDUFrame(0, (0, 0, 8)), PortPolarity.STUD))
            p_list.append(PortSpec(3, LDUFrame(0, (0, 20, 8)), PortPolarity.STUD))
        p_list.append(
            PortSpec(
                len(p_list),
                LDUFrame(ROT_Z_TO_POS_X if pol == PortPolarity.STUD else ROT_Z_TO_NEG_X, (16, 10, 4)),
                pol,
                ctype,
            )
        )
        _reg(
            BrickPrimitive(
                part_id=hpid,
                name=hname,
                hull_boxes=(LDUBox(-10, -10, 0, 10, 30, 8),),
                ports=tuple(p_list),
                mass_grams=0.40,
                is_snot=True,
            )
        )
    _reg(
        BrickPrimitive(
            part_id="2921",
            name="Brick Special 1x1 with Handle",
            hull_boxes=(LDUBox(-10, -10, 0, 10, 10, 24),),
            ports=(
                PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
                PortSpec(1, LDUFrame(0, (0, 0, 24)), PortPolarity.STUD),
                PortSpec(2, LDUFrame(ROT_Z_TO_POS_Y, (16, 0, 12)), PortPolarity.STUD, ConnectorType.BAR_3_18MM),
            ),
            mass_grams=0.50,
            is_snot=True,
        )
    )

    return cat


# Canonical mold-variant aliases used in LDraw `.mpd` / `.ldr` files
LDRAW_PART_ALIASES: Dict[str, str] = {
    "3040": "3040b",
    "3040a": "3040b",
    "3068": "3068b",
    "3068a": "3068b",
    "3069": "3069b",
    "3069a": "3069b",
    "3070": "3070b",
    "3070a": "3070b",
    "2412": "2412b",
    "2412a": "2412b",
    "3062": "3062b",
    "3062a": "3062b",
    "3245": "3245c",
    "3245a": "3245c",
    "3245b": "3245c",
    "3794": "3794a",
    "3794b": "3794a",
    "4032": "4032a",
    "4032b": "4032a",
    "4081": "4081b",
    "4081a": "4081b",
    "4085": "4085d",
    "4085a": "4085d",
    "4085b": "4085d",
    "4085c": "4085d",
    "3957": "3957b",
    "3957a": "3957b",
    "4287": "4287c",
    "4287a": "4287c",
    "4287b": "4287c",
    "30367": "30367c",
    "30367a": "30367c",
    "30367b": "30367c",
    "30151": "30151b",
    "30151a": "30151b",
    "3942": "3942c",
    "3942a": "3942c",
    "3942b": "3942c",
    "6014": "6014b",
    "6014a": "6014b",
    "3665a": "3665",
    "3665b": "3665",
    "32064": "32064a",
    "32064b": "32064a",
    "32123": "32123a",
    "32123b": "32123a",
    "98138": "98138pr0008",
    "59900b": "59900",
    "6141p01": "6141",
    "2429c01": "2429",
    "3680c01": "3680",
    "3937c01": "3937",
    "4865": "4865b",
    "4865a": "4865b",
    "6231": "91501",
    "25269": "3070b",
    "4265a": "32123a",
    "4265b": "32123a",
    "4265c": "32123a",
    "3651": "6536",
    "62462": "6536",
    "4073": "6141",
    "85861": "6141",
    "24866": "33291",
    "30008": "3024",
    "50746": "54200",
    "50967": "54200",
    "57878": "60593",
    "6143": "3941",
    "3044": "3044b",
    "3044a": "3044b",
    "30055": "15332",
    "56145": "55981",
    "44309": "58090",
    "3641": "87697",
    "3712": "3679",
    "6134": "3938",
    "30377": "30374",
    "48336": "2540",
    "6587": "4519",
    "3749": "43093",
    "3704": "32062",
    "4273b": "6536",
    "4275b": "60471",
    "44302a": "60471",
    "44301a": "60471",
    "44676": "2555",
    "6264": "3062b",
    "54086": "3039",
    "2878": "3001",
    "57877": "60592",
    "73129": "3941",
    "3326b": "3937",
}

_PRINT_SUFFIX_RE = re.compile(r"^([0-9]+[a-z]?)(?:p[a-z0-9]+|d[0-9]+)$")
_OMR_PREFIX_RE = re.compile(r"^[0-9a-z_-]+\s+-\s+(.+)$")


def normalize_ldraw_part_id(
    raw_token: str,
    catalog: Optional[Dict[str, BrickPrimitive]] = None,
) -> str:
    """
    Normalize a raw LDraw part reference (e.g., `3004.dat`, `parts/3069bp01.dat`, `3040a`,
    `21035 - 3623.dat`) into a canonical catalog `part_id` when supported by `catalog` or `LDRAW_PART_ALIASES`.
    """
    tok = raw_token.strip().lower().replace("\\", "/")
    if tok.endswith(".dat"):
        tok = tok[:-4]
    if "/" in tok:
        tok = tok.split("/")[-1]
    m_omr = _OMR_PREFIX_RE.match(tok)
    if m_omr:
        tok = m_omr.group(1).strip()
    if catalog is not None and tok in catalog:
        return tok
    if tok in LDRAW_PART_ALIASES:
        mapped = LDRAW_PART_ALIASES[tok]
        if catalog is None or mapped in catalog:
            return mapped
    m = _PRINT_SUFFIX_RE.match(tok)
    if m:
        base = m.group(1)
        if catalog is not None and base in catalog:
            return base
        if base in LDRAW_PART_ALIASES:
            mapped_b = LDRAW_PART_ALIASES[base]
            if catalog is None or mapped_b in catalog:
                return mapped_b
    return tok
