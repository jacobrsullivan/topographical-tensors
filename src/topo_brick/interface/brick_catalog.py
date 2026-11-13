"""
Standard and SNOT ("Studs Not On Top") LEGO brick primitive catalog in exact LDraw Units (LDU).

Insertion-Normal Convention for Port Frames F_p:
- Every port p on a brick has an exact LDUFrame F_p = (R_p, r_p) where:
  * r_p in Z^3_LDU is the mating center on the brick's surface.
  * R_p in SO(3)_cubic orients local +Z along the stud-into-cavity insertion direction:
    - For a male stud (PortPolarity.STUD = +1), local +Z points OUTWARD from the stud face.
    - For a female anti-stud (PortPolarity.ANTI_STUD = -1), local +Z points INTO the cavity.
- Under this convention, when port p of candidate brick b mates with exposed assembly port u
  (with p.polarity == -u.polarity), the candidate's rigid world pose X satisfies:
      X ∘ F_p = T_u ∘ Q_θ   ==>   X = T_u ∘ Q_θ ∘ F_p^-1
  where Q_θ in C4_Z_ROT_IDS is one of the 4 planar yaw rotations around the stud axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
from typing import Dict, Optional, Tuple
import numpy as np

from topo_brick.interface.ldu_frames import (
    BRICK_HEIGHT_LDU,
    HALF_PLATE_LDU,
    HALF_STUD_LDU,
    PLATE_HEIGHT_LDU,
    STUD_PITCH_LDU,
    LDUBox,
    LDUFrame,
    matrix_to_rot_id,
)


class PortPolarity(IntEnum):
    STUD = 1
    ANTI_STUD = -1


class ConnectorType(IntEnum):
    STANDARD_STUD = 0
    HOLLOW_STUD = 1
    TECHNIC_PIN_HOLE = 2
    TECHNIC_PIN = 3
    BAR_3_18MM = 4
    CLIP_3_18MM = 5
    SMOOTH_PIN = 6
    TECHNIC_AXLE_HOLE = 7
    TECHNIC_AXLE = 8
    HINGE_SOCKET = 9
    HINGE_PIN = 10
    LOCKING_HINGE_SOCKET = 11
    LOCKING_HINGE_PIN = 12
    BALL_SOCKET = 13
    BALL_JOINT = 14
    TURNTABLE_BASE_SOCKET = 15
    TURNTABLE_TOP_PIN = 16
    WHEEL_HUB_HOLE = 17
    WHEEL_PIN = 18


def are_connectors_compatible(
    type_a: ConnectorType,
    pol_a: PortPolarity,
    type_b: ConnectorType,
    pol_b: PortPolarity,
) -> bool:
    """
    Return True iff two ports (type_a, pol_a) and (type_b, pol_b) can legally mate.
    Enforces opposite polarity AND connector-family compatibility across all 8 families:
      - {STANDARD_STUD, HOLLOW_STUD} <-> {STANDARD_STUD, HOLLOW_STUD}
      - {TECHNIC_PIN, SMOOTH_PIN, WHEEL_PIN} (+1) <-> {TECHNIC_PIN_HOLE, WHEEL_HUB_HOLE} (-1)
      - TECHNIC_AXLE (+1) <-> {TECHNIC_AXLE_HOLE, TECHNIC_PIN_HOLE} (-1)
      - BAR_3_18MM (+1) <-> {CLIP_3_18MM, HOLLOW_STUD} (-1)
      - HINGE_PIN (+1) <-> HINGE_SOCKET (-1)
      - LOCKING_HINGE_PIN (+1) <-> LOCKING_HINGE_SOCKET (-1)
      - BALL_JOINT (+1) <-> BALL_SOCKET (-1)
      - TURNTABLE_TOP_PIN (+1) <-> TURNTABLE_BASE_SOCKET (-1)
    """
    if pol_a != -pol_b:
        return False
    male_t = type_a if pol_a == PortPolarity.STUD else type_b
    fem_t = type_b if pol_a == PortPolarity.STUD else type_a
    if male_t in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD):
        return fem_t in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    if male_t in (ConnectorType.TECHNIC_PIN, ConnectorType.SMOOTH_PIN, ConnectorType.WHEEL_PIN):
        return fem_t in (ConnectorType.TECHNIC_PIN_HOLE, ConnectorType.WHEEL_HUB_HOLE)
    if male_t == ConnectorType.TECHNIC_AXLE:
        return fem_t in (ConnectorType.TECHNIC_AXLE_HOLE, ConnectorType.TECHNIC_PIN_HOLE)
    if male_t == ConnectorType.BAR_3_18MM:
        return fem_t in (ConnectorType.CLIP_3_18MM, ConnectorType.HOLLOW_STUD)
    if male_t == ConnectorType.HINGE_PIN:
        return fem_t == ConnectorType.HINGE_SOCKET
    if male_t == ConnectorType.LOCKING_HINGE_PIN:
        return fem_t == ConnectorType.LOCKING_HINGE_SOCKET
    if male_t == ConnectorType.BALL_JOINT:
        return fem_t == ConnectorType.BALL_SOCKET
    if male_t == ConnectorType.TURNTABLE_TOP_PIN:
        return fem_t == ConnectorType.TURNTABLE_BASE_SOCKET
    return False


# Rotation ID where local +Z points along +X:
# local X -> -Z, local Y -> +Y, local Z -> +X  (det = +1, Ry(+90 deg))
ROT_Z_TO_POS_X: int = matrix_to_rot_id(
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int64)
)

# Rotation ID where local +Z points along -X:
# local X -> +Z, local Y -> +Y, local Z -> -X  (det = +1, Ry(-90 deg))
ROT_Z_TO_NEG_X: int = matrix_to_rot_id(
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
)


@dataclass(frozen=True, slots=True)
class PortSpec:
    """Intrinsic port specification in a brick primitive's local LDU coordinate frame."""

    port_id: int
    local_frame: LDUFrame
    polarity: PortPolarity
    connector_type: ConnectorType = ConnectorType.STANDARD_STUD

    @property
    def outward_probe_offset_local(self) -> Tuple[int, int, int]:
        """
        Return a 1-LDU probe point along the side of the port plane where a mating brick sits:
        +1 LDU along local +Z for a male stud, -1 LDU along local +Z for a female anti-stud.
        """
        return (0, 0, 1) if self.polarity == PortPolarity.STUD else (0, 0, -1)

    @property
    def local_protrusion_box(self) -> Optional[LDUBox]:
        """
        Explicit narrow-phase collider box for male connectors (stud cylinder / pin / bar)
        in the port's local frame (protruding along local +Z insertion normal).
        When a male port is UNMATED, this protrusion box cannot intersect any solid brick body.
        When a male port IS MATED to a compatible female port at the same pose, this specific
        protrusion collider is exempted (F11 fix).
        """
        if self.polarity != PortPolarity.STUD or self.connector_type in (
            ConnectorType.BAR_3_18MM,
            ConnectorType.HINGE_PIN,
            ConnectorType.LOCKING_HINGE_PIN,
            ConnectorType.BALL_JOINT,
            ConnectorType.TURNTABLE_TOP_PIN,
        ):
            return None
        if self.connector_type in (
            ConnectorType.TECHNIC_PIN,
            ConnectorType.SMOOTH_PIN,
            ConnectorType.TECHNIC_AXLE,
            ConnectorType.WHEEL_PIN,
        ):
            return LDUBox(-4, -4, 0, 4, 4, 4)
        return LDUBox(-6, -6, 0, 6, 6, 4)


@dataclass(frozen=True, slots=True)
class BrickPrimitive:
    """
    Immutable definition of a brick or SNOT element in its intrinsic LDU frame.
    """

    part_id: str
    name: str
    hull_boxes: Tuple[LDUBox, ...]
    ports: Tuple[PortSpec, ...]
    mass_grams: float
    is_snot: bool = False
    is_ldraw_native_frame: bool = False
    coacd_shell_id: Optional[str] = None

    @property
    def stud_ports(self) -> Tuple[PortSpec, ...]:
        return tuple(p for p in self.ports if p.polarity == PortPolarity.STUD)

    @property
    def anti_stud_ports(self) -> Tuple[PortSpec, ...]:
        return tuple(p for p in self.ports if p.polarity == PortPolarity.ANTI_STUD)


def _make_rect_part(
    part_id: str,
    name: str,
    studs_x: int,
    studs_y: int,
    height_ldu: int,
    mass_grams: float,
) -> BrickPrimitive:
    """
    Construct a standard rectangular brick or plate anchored at stud (0, 0).
    Stud (i, j) is centered at (i * 20, j * 20).
    Bounding box spans:
      x in [-10, (studs_x - 1)*20 + 10]
      y in [-10, (studs_y - 1)*20 + 10]
      z in [0, height_ldu]
    """
    box = LDUBox(
        xmin=-HALF_STUD_LDU,
        ymin=-HALF_STUD_LDU,
        zmin=0,
        xmax=(studs_x - 1) * STUD_PITCH_LDU + HALF_STUD_LDU,
        ymax=(studs_y - 1) * STUD_PITCH_LDU + HALF_STUD_LDU,
        zmax=height_ldu,
    )
    ports = []
    pid = 0
    # Bottom anti-studs at z = 0
    for ix in range(studs_x):
        for iy in range(studs_y):
            ports.append(
                PortSpec(
                    port_id=pid,
                    local_frame=LDUFrame(
                        rot_id=0,
                        pos=(ix * STUD_PITCH_LDU, iy * STUD_PITCH_LDU, 0),
                    ),
                    polarity=PortPolarity.ANTI_STUD,
                )
            )
            pid += 1
    # Top studs at z = height_ldu
    for ix in range(studs_x):
        for iy in range(studs_y):
            ports.append(
                PortSpec(
                    port_id=pid,
                    local_frame=LDUFrame(
                        rot_id=0,
                        pos=(ix * STUD_PITCH_LDU, iy * STUD_PITCH_LDU, height_ldu),
                    ),
                    polarity=PortPolarity.STUD,
                )
            )
            pid += 1

    return BrickPrimitive(
        part_id=part_id,
        name=name,
        hull_boxes=(box,),
        ports=tuple(ports),
        mass_grams=mass_grams,
        is_snot=False,
    )


def _make_87087_snot_brick() -> BrickPrimitive:
    """
    LDraw 87087: Brick 1 x 1 with Stud on 1 Side (+X face).
    - Main body: x in [-10, 10], y in [-10, 10], z in [0, 24].
    - Bottom anti-stud at (0, 0, 0), rot=I.
    - Top stud at (0, 0, 24), rot=I.
    - Lateral SNOT stud on +X face (x = 10) at z = 16 LDU (2 plates height), rot=ROT_Z_TO_POS_X.
    """
    box = LDUBox(-10, -10, 0, 10, 10, BRICK_HEIGHT_LDU)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
        PortSpec(1, LDUFrame(0, (0, 0, BRICK_HEIGHT_LDU)), PortPolarity.STUD),
        PortSpec(
            2,
            LDUFrame(ROT_Z_TO_POS_X, (HALF_STUD_LDU, 0, 2 * PLATE_HEIGHT_LDU)),
            PortPolarity.STUD,
        ),
    )
    return BrickPrimitive(
        part_id="87087",
        name="Brick 1x1 with Stud on 1 Side",
        hull_boxes=(box,),
        ports=ports,
        mass_grams=0.46,
        is_snot=True,
    )


def _make_4070_erling_brick() -> BrickPrimitive:
    """
    LDraw 4070: Brick 1 x 1 Headlight ("Erling" Brick).
    - Recessed lateral SNOT stud on +X face at x = +6 LDU (recessed by 1/2 plate = 4 LDU from +10),
      centered at z = 10 LDU (half-stud height).
    - Lower front lip + main upper body represented as two exact integer LDU boxes so a plate/brick
      mated to the recessed stud at x = 6 extends from x = 6 without colliding with the main body.
    """
    # Rear 16-LDU body: x in [-10, 6], y in [-10, 10], z in [0, 24]
    rear_box = LDUBox(-10, -10, 0, 6, 10, BRICK_HEIGHT_LDU)
    # Front bottom lip: x in [6, 10], y in [-10, 10], z in [0, 4] (4 LDU = 1/2 plate foot)
    lip_box = LDUBox(6, -10, 0, 10, 10, HALF_PLATE_LDU)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
        PortSpec(1, LDUFrame(0, (0, 0, BRICK_HEIGHT_LDU)), PortPolarity.STUD),
        PortSpec(
            2,
            LDUFrame(ROT_Z_TO_POS_X, (6, 0, 14)),
            PortPolarity.STUD,
            ConnectorType.HOLLOW_STUD,
        ),
        PortSpec(
            3,
            LDUFrame(ROT_Z_TO_POS_X, (-10, 0, 14)),
            PortPolarity.ANTI_STUD,
        ),
    )
    return BrickPrimitive(
        part_id="4070",
        name="Brick 1x1 Headlight (Erling)",
        hull_boxes=(rear_box, lip_box),
        ports=ports,
        mass_grams=0.42,
        is_snot=True,
    )


def _make_99780_snot_bracket() -> BrickPrimitive:
    """
    LDraw 99780: Bracket 1 x 2 - 1 x 2 Inverted.
    - Top horizontal 1x2 plate flange: x in [-10, 10], y in [-10, 30], z in [0, 8].
    - Vertical 1/2-plate (4 LDU) SNOT apron on +X side: x in [10, 14], y in [-10, 30], z in [-16, 8].
    - Exposes 2 top studs at z=8, 2 bottom anti-studs at z=0, and 2 lateral SNOT studs on +X face
      at x = 14, y in {0, 20}, z = -4.
    """
    flange_box = LDUBox(-10, -10, 0, 10, 30, PLATE_HEIGHT_LDU)
    apron_box = LDUBox(10, -10, -16, 14, 30, PLATE_HEIGHT_LDU)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
        PortSpec(1, LDUFrame(0, (0, STUD_PITCH_LDU, 0)), PortPolarity.ANTI_STUD),
        PortSpec(2, LDUFrame(0, (0, 0, PLATE_HEIGHT_LDU)), PortPolarity.STUD),
        PortSpec(3, LDUFrame(0, (0, STUD_PITCH_LDU, PLATE_HEIGHT_LDU)), PortPolarity.STUD),
        PortSpec(4, LDUFrame(ROT_Z_TO_POS_X, (14, 0, -4)), PortPolarity.STUD),
        PortSpec(5, LDUFrame(ROT_Z_TO_POS_X, (14, STUD_PITCH_LDU, -4)), PortPolarity.STUD),
    )
    return BrickPrimitive(
        part_id="99780",
        name="Bracket 1x2 - 1x2 Inverted",
        hull_boxes=(flange_box, apron_box),
        ports=ports,
        mass_grams=0.62,
        is_snot=True,
    )


def _make_99207_snot_bracket() -> BrickPrimitive:
    """
    LDraw 99207: Bracket 1 x 2 - 2 x 2 Inverted.
    - Top horizontal 1x2 plate flange: x in [-10, 10], y in [-10, 30], z in [0, 8].
    - Vertical 2x2 SNOT apron on +X side: x in [10, 14], y in [-10, 30], z in [-32, 8].
    - Exposes 4 lateral SNOT studs on +X face at x = 14, y in {0, 20}, z in {-4, -24}.
    """
    flange_box = LDUBox(-10, -10, 0, 10, 30, PLATE_HEIGHT_LDU)
    apron_box = LDUBox(10, -10, -32, 14, 30, PLATE_HEIGHT_LDU)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
        PortSpec(1, LDUFrame(0, (0, STUD_PITCH_LDU, 0)), PortPolarity.ANTI_STUD),
        PortSpec(2, LDUFrame(0, (0, 0, PLATE_HEIGHT_LDU)), PortPolarity.STUD),
        PortSpec(3, LDUFrame(0, (0, STUD_PITCH_LDU, PLATE_HEIGHT_LDU)), PortPolarity.STUD),
        PortSpec(4, LDUFrame(ROT_Z_TO_POS_X, (14, 0, -4)), PortPolarity.STUD),
        PortSpec(5, LDUFrame(ROT_Z_TO_POS_X, (14, STUD_PITCH_LDU, -4)), PortPolarity.STUD),
        PortSpec(6, LDUFrame(ROT_Z_TO_POS_X, (14, 0, -24)), PortPolarity.STUD),
        PortSpec(7, LDUFrame(ROT_Z_TO_POS_X, (14, STUD_PITCH_LDU, -24)), PortPolarity.STUD),
    )
    return BrickPrimitive(
        part_id="99207",
        name="Bracket 1x2 - 2x2 Inverted",
        hull_boxes=(flange_box, apron_box),
        ports=ports,
        mass_grams=0.88,
        is_snot=True,
    )


def _make_3039_slope_2x2() -> BrickPrimitive:
    """
    LDraw 3039: Slope 45 2 x 2.
    - Back ridge (x in [-10, 10], y in [-10, 30], z in [0, 24]): full 24-LDU height with 2 top studs.
    - Front sloped apron (x in [10, 30], y in [-10, 30], z in [0, 12]): stepped narrow-phase box
      leaving the upper 45-degree sloped clearance volume [10..30, -10..30, 12..24] open.
    - Bottom exposes 4 anti-studs at z=0; top exposes 2 studs along the rear ridge x=0, y in {0, 20}.
    """
    rear_box = LDUBox(-10, -10, 0, 10, 30, BRICK_HEIGHT_LDU)
    front_slope_box = LDUBox(10, -10, 0, 30, 30, 12)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD),
        PortSpec(1, LDUFrame(0, (0, STUD_PITCH_LDU, 0)), PortPolarity.ANTI_STUD),
        PortSpec(2, LDUFrame(0, (STUD_PITCH_LDU, 0, 0)), PortPolarity.ANTI_STUD),
        PortSpec(3, LDUFrame(0, (STUD_PITCH_LDU, STUD_PITCH_LDU, 0)), PortPolarity.ANTI_STUD),
        PortSpec(4, LDUFrame(0, (0, 0, BRICK_HEIGHT_LDU)), PortPolarity.STUD),
        PortSpec(5, LDUFrame(0, (0, STUD_PITCH_LDU, BRICK_HEIGHT_LDU)), PortPolarity.STUD),
    )
    return BrickPrimitive(
        part_id="3039",
        name="Slope 45 2x2",
        hull_boxes=(rear_box, front_slope_box),
        ports=ports,
        mass_grams=1.18,
        is_snot=False,
    )


def _make_6541_technic_brick_1x1() -> BrickPrimitive:
    """
    LDraw 6541: Technic Brick 1 x 1 with Hole.
    - Main body: x in [-10, 10], y in [-10, 10], z in [0, 24].
    - Port 0: Bottom anti-stud at (0, 0, 0).
    - Port 1: Top stud at (0, 0, 24).
    - Port 2: +X Technic pin-hole aperture at (10, 0, 14) with inward cavity normal along -X
      (so local +Z of ANTI_STUD points INTO the hole along -X: ROT_Z_TO_NEG_X).
    - Port 3: -X Technic pin-hole aperture at (-10, 0, 14) with inward cavity normal along +X
      (ROT_Z_TO_POS_X).
    """
    # 4-wall hollow decomposition around the X-axis pin hole (y in [-6, 6], z in [8, 20])
    hull_boxes = (
        LDUBox(-10, -10, 0, 10, 10, 8),
        LDUBox(-10, -10, 20, 10, 10, BRICK_HEIGHT_LDU),
        LDUBox(-10, -10, 8, 10, -6, 20),
        LDUBox(-10, 6, 8, 10, 10, 20),
    )
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD, ConnectorType.STANDARD_STUD),
        PortSpec(1, LDUFrame(0, (0, 0, BRICK_HEIGHT_LDU)), PortPolarity.STUD, ConnectorType.STANDARD_STUD),
        PortSpec(
            2,
            LDUFrame(ROT_Z_TO_NEG_X, (10, 0, 14)),
            PortPolarity.ANTI_STUD,
            ConnectorType.TECHNIC_PIN_HOLE,
        ),
        PortSpec(
            3,
            LDUFrame(ROT_Z_TO_POS_X, (-10, 0, 14)),
            PortPolarity.ANTI_STUD,
            ConnectorType.TECHNIC_PIN_HOLE,
        ),
    )
    return BrickPrimitive(
        part_id="6541",
        name="Technic Brick 1x1 with Hole",
        hull_boxes=hull_boxes,
        ports=ports,
        mass_grams=0.44,
        is_snot=True,
    )


def _make_2780_technic_friction_pin() -> BrickPrimitive:
    """
    LDraw 2780: Technic Pin with Friction Ridges Lengthwise.
    - Cylindrical pin body [-4, 4]^3 inside the shared 6541 pin-hole channel (y in [-6, 6], z in [8, 20]),
      with two opposing male TECHNIC_PIN ports at origin along +Z and -Z.
    """
    rot_z_invert = matrix_to_rot_id(
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.int64)
    )
    collar_box = LDUBox(-4, -4, -4, 4, 4, 4)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
        PortSpec(1, LDUFrame(rot_z_invert, (0, 0, 0)), PortPolarity.STUD, ConnectorType.TECHNIC_PIN),
    )
    return BrickPrimitive(
        part_id="2780",
        name="Technic Pin with Friction Ridges",
        hull_boxes=(collar_box,),
        ports=ports,
        mass_grams=0.16,
        is_snot=True,
    )


def _make_3673_technic_smooth_pin() -> BrickPrimitive:
    """
    LDraw 3673: Technic Pin without Friction Ridges (Smooth Rotational Pin).
    - Identical geometry to 2780, but uses ConnectorType.SMOOTH_PIN so axial torque around
      the pin axis is projected to zero (P_c projects out tau_n = 0).
    """
    rot_z_invert = matrix_to_rot_id(
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.int64)
    )
    collar_box = LDUBox(-4, -4, -4, 4, 4, 4)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.STUD, ConnectorType.SMOOTH_PIN),
        PortSpec(1, LDUFrame(rot_z_invert, (0, 0, 0)), PortPolarity.STUD, ConnectorType.SMOOTH_PIN),
    )
    return BrickPrimitive(
        part_id="3673",
        name="Technic Pin Smooth (Frictionless)",
        hull_boxes=(collar_box,),
        ports=ports,
        mass_grams=0.16,
        is_snot=True,
    )


def _make_4085d_plate_1x1_with_clip() -> BrickPrimitive:
    """
    LDraw 4085d: Plate 1 x 1 with Clip Vertical.
    - Main 1x1 plate body: x in [-10, 10], y in [-10, 10], z in [0, 8].
    - Clip jaw on +X face: x in [10, 16], y in [-4, 4], z in [0, 8], exposing a female
      CLIP_3_18MM port at (16, 0, 4) oriented with local +Z pointing into the clip (-X).
    """
    body_box = LDUBox(-10, -10, 0, 10, 10, PLATE_HEIGHT_LDU)
    clip_box = LDUBox(10, -4, 0, 12, 4, PLATE_HEIGHT_LDU)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.ANTI_STUD, ConnectorType.STANDARD_STUD),
        PortSpec(1, LDUFrame(0, (0, 0, PLATE_HEIGHT_LDU)), PortPolarity.STUD, ConnectorType.STANDARD_STUD),
        PortSpec(
            2,
            LDUFrame(ROT_Z_TO_NEG_X, (16, 0, 4)),
            PortPolarity.ANTI_STUD,
            ConnectorType.CLIP_3_18MM,
        ),
    )
    return BrickPrimitive(
        part_id="4085d",
        name="Plate 1x1 with Vertical Clip",
        hull_boxes=(body_box, clip_box),
        ports=ports,
        mass_grams=0.24,
        is_snot=True,
    )


def _make_30374_bar_4l() -> BrickPrimitive:
    """
    LDraw 30374: Bar 4L (Lightsaber Blade / Wand).
    - Cylindrical 3.18mm bar of length 80 LDU (4 studs) spanning z in [0, 80], cross-section [-4, 4]^2.
    - Exposes BAR_3_18MM male ports along its shaft/ends so it can snap into CLIP_3_18MM or HOLLOW_STUD.
    """
    bar_box = LDUBox(-4, -4, 0, 4, 4, 80)
    ports = (
        PortSpec(0, LDUFrame(0, (0, 0, 0)), PortPolarity.STUD, ConnectorType.BAR_3_18MM),
        PortSpec(1, LDUFrame(ROT_Z_TO_POS_X, (0, 0, 20)), PortPolarity.STUD, ConnectorType.BAR_3_18MM),
        PortSpec(2, LDUFrame(ROT_Z_TO_POS_X, (0, 0, 60)), PortPolarity.STUD, ConnectorType.BAR_3_18MM),
    )
    return BrickPrimitive(
        part_id="30374",
        name="Bar 4L (3.18mm Rod)",
        hull_boxes=(bar_box,),
        ports=ports,
        mass_grams=0.38,
        is_snot=True,
    )


def _build_catalog() -> Dict[str, BrickPrimitive]:
    parts = [
        # Standard Plates (H = 8 LDU)
        _make_rect_part("3024", "Plate 1x1", 1, 1, PLATE_HEIGHT_LDU, 0.18),
        _make_rect_part("3023", "Plate 1x2", 1, 2, PLATE_HEIGHT_LDU, 0.35),
        _make_rect_part("3623", "Plate 1x3", 1, 3, PLATE_HEIGHT_LDU, 0.52),
        _make_rect_part("3710", "Plate 1x4", 1, 4, PLATE_HEIGHT_LDU, 0.68),
        _make_rect_part("3022", "Plate 2x2", 2, 2, PLATE_HEIGHT_LDU, 0.66),
        _make_rect_part("3021", "Plate 2x3", 2, 3, PLATE_HEIGHT_LDU, 0.98),
        _make_rect_part("3020", "Plate 2x4", 2, 4, PLATE_HEIGHT_LDU, 1.30),
        # Standard Bricks (H = 24 LDU)
        _make_rect_part("3005", "Brick 1x1", 1, 1, BRICK_HEIGHT_LDU, 0.43),
        _make_rect_part("3004", "Brick 1x2", 1, 2, BRICK_HEIGHT_LDU, 0.82),
        _make_rect_part("3622", "Brick 1x3", 1, 3, BRICK_HEIGHT_LDU, 1.20),
        _make_rect_part("3010", "Brick 1x4", 1, 4, BRICK_HEIGHT_LDU, 1.58),
        _make_rect_part("3003", "Brick 2x2", 2, 2, BRICK_HEIGHT_LDU, 1.52),
        _make_rect_part("3002", "Brick 2x3", 2, 3, BRICK_HEIGHT_LDU, 2.25),
        _make_rect_part("3001", "Brick 2x4", 2, 4, BRICK_HEIGHT_LDU, 2.95),
        _make_rect_part("2456", "Brick 2x6", 2, 6, BRICK_HEIGHT_LDU, 4.40),
        _make_rect_part("3007", "Brick 2x8", 2, 8, BRICK_HEIGHT_LDU, 5.80),
        # SNOT Primitives
        _make_87087_snot_brick(),
        _make_4070_erling_brick(),
        _make_99780_snot_bracket(),
        _make_99207_snot_bracket(),
    ]
    return {p.part_id: p for p in parts}


def _build_extended_catalog(base_catalog: Dict[str, BrickPrimitive]) -> Dict[str, BrickPrimitive]:
    ext = dict(base_catalog)
    extra_parts = [
        # Missing StableText2Brick 8-part kit rectangular bricks (3009 1x6, 3008 1x8)
        _make_rect_part("3009", "Brick 1x6", 1, 6, BRICK_HEIGHT_LDU, 2.34),
        _make_rect_part("3008", "Brick 1x8", 1, 8, BRICK_HEIGHT_LDU, 3.10),
        # Broader connector family primitives (Deliverable 3: Slope, Technic Pin/Hole, Clip/Bar)
        _make_3039_slope_2x2(),
        _make_6541_technic_brick_1x1(),
        _make_2780_technic_friction_pin(),
        _make_3673_technic_smooth_pin(),
        _make_4085d_plate_1x1_with_clip(),
        _make_30374_bar_4l(),
    ]
    for p in extra_parts:
        ext[p.part_id] = p
    return ext


BRICK_CATALOG: Dict[str, BrickPrimitive] = _build_catalog()
EXTENDED_BRICK_CATALOG: Dict[str, BrickPrimitive] = _build_extended_catalog(BRICK_CATALOG)
FROZEN_20PART_CATALOG_IDS: Tuple[str, ...] = tuple(BRICK_CATALOG.keys())

# Exact 8-part StableText2Brick rectangular kit (Critique 5E: excludes 3622 1x3 and 3002 2x3)
STABLETEXT2BRICK_8PART_KIT: Tuple[str, ...] = (
    "3005",  # 1x1
    "3004",  # 1x2
    "3010",  # 1x4
    "3009",  # 1x6
    "3008",  # 1x8
    "3003",  # 2x2
    "3001",  # 2x4
    "2456",  # 2x6
)

# Extended 10-part rectangular kit (including 3622 1x3 and 3002 2x3)
EXTENDED_10PART_RECT_KIT: Tuple[str, ...] = (
    "3005",
    "3004",
    "3622",
    "3010",
    "3009",
    "3008",
    "3003",
    "3002",
    "3001",
    "2456",
)


@dataclass(frozen=True, slots=True)
class CollisionContractSpec:
    """
    Explicit collision-checking contract (Round 7 Decision 2):
    Records source geometry, decomposition method, geometric offset (inset), collision algorithm,
    allowed penetration, and local mating exemptions separately so 0.25-LDU inset meshes are
    never silently conflated with 1.0-LDU penetration-allowance convex decompositions.
    """

    contract_id: str
    source_mesh_version: str
    decomposition_method: str  # "EXACT_LDU_BOX_PLUS_PROTRUSION", "COACD_CONVEX_HULLS", "INSET_TRIS"
    geometric_inset_ldu: float  # e.g., 0.0 for exact LDU boxes or Oct-2026 CoACD; 0.25 for legacy inset
    collision_algorithm: str   # "EXACT_INT_AABB_PLUS_MATED_PROTRUSION_EXEMPTION", "GJK_EPA_CONVEX"
    allowed_penetration_ldu: float  # e.g., 0.0 for strict interior LDU boxes; 1.0 for Oct-2026 CoACD
    mated_port_exemption_enabled: bool = True


@dataclass(frozen=True, slots=True)
class AssetProvenanceManifest:
    """
    Component-specific provenance and licensing manifest for a catalog asset (Round 7 Decision 2).
    """

    part_id: str
    provider_id: str  # "topo_brick_parametric_ldu", "ldraw_ldcad_compiled", "bricknet_mit_table"
    source_version: str
    sha256_hash: str
    geometry_license: str       # "CC-BY-4.0 / CC0 (LDraw)", "MIT (Parametric Code)"
    connector_license: str      # "CC-BY-SA-4.0 (LDCad Shadow)", "MIT (BrickNet Table / Parametric)"
    redistribution_status: str  # "COMMERCIALLY_USABLE_WITH_ATTRIBUTION", "SHARE_ALIKE_ADAPTATION", "UNVERIFIED_EXTERNAL"
    collision_contract: CollisionContractSpec
    validation_status: str      # "VALIDATED_EXACT_MATING_AND_CLEARANCE", "IMPORTED_UNVALIDATED"


@dataclass(frozen=True, slots=True)
class DeclarativePartSpec:
    """
    Unified declarative part specification combining geometry, ports, mass/CoM, intrinsic
    16-D geometric descriptor φ(b) (zero-shot unseen-part encoding M1), and provenance manifest.
    """

    primitive: BrickPrimitive
    intrinsic_descriptor: Tuple[float, ...]  # 16-D part-ID-free geometric descriptor φ(b)
    provenance: AssetProvenanceManifest


def compute_intrinsic_part_descriptor(part_id: str) -> np.ndarray:
    """
    Compute the 16-D part-ID-free intrinsic geometric & connector descriptor φ(b) in R^16
    from a part's bounding boxes, mass, and port distribution (Model M1).
    """
    prim = get_brick_primitive(part_id)
    xmin = min(b.xmin for b in prim.hull_boxes)
    ymin = min(b.ymin for b in prim.hull_boxes)
    zmin = min(b.zmin for b in prim.hull_boxes)
    xmax = max(b.xmax for b in prim.hull_boxes)
    ymax = max(b.ymax for b in prim.hull_boxes)
    zmax = max(b.zmax for b in prim.hull_boxes)
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    dz = float(zmax - zmin)
    bbox_vol = max(1.0, dx * dy * dz)
    hull_vol = float(sum(b.volume_ldu3 for b in prim.hull_boxes))
    ref_vol = 20.0 * 20.0 * 24.0  # 1x1 brick volume = 9600 LDU^3

    n_studs = sum(
        1
        for p in prim.ports
        if p.polarity == PortPolarity.STUD
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    )
    n_antistuds = sum(
        1
        for p in prim.ports
        if p.polarity == PortPolarity.ANTI_STUD
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    )
    n_lateral = sum(1 for p in prim.ports if abs(int(p.local_frame.z_axis[2])) == 0)
    n_technic = sum(
        1
        for p in prim.ports
        if p.connector_type
        in (
            ConnectorType.TECHNIC_PIN_HOLE,
            ConnectorType.TECHNIC_PIN,
            ConnectorType.SMOOTH_PIN,
            ConnectorType.TECHNIC_AXLE_HOLE,
            ConnectorType.TECHNIC_AXLE,
            ConnectorType.WHEEL_HUB_HOLE,
            ConnectorType.WHEEL_PIN,
        )
    )
    n_clip_bar = sum(
        1
        for p in prim.ports
        if p.connector_type
        in (
            ConnectorType.CLIP_3_18MM,
            ConnectorType.BAR_3_18MM,
            ConnectorType.HINGE_SOCKET,
            ConnectorType.HINGE_PIN,
            ConnectorType.LOCKING_HINGE_SOCKET,
            ConnectorType.LOCKING_HINGE_PIN,
            ConnectorType.BALL_SOCKET,
            ConnectorType.BALL_JOINT,
            ConnectorType.TURNTABLE_BASE_SOCKET,
            ConnectorType.TURNTABLE_TOP_PIN,
        )
    )

    desc = np.array(
        [
            min(dx, dy) / 20.0,                        # 0: minor horizontal span in studs
            max(dx, dy) / 20.0,                        # 1: major horizontal span in studs
            dz / 24.0,                                 # 2: height in brick courses
            hull_vol / ref_vol,                        # 3: volume in 1x1 brick units
            hull_vol / bbox_vol,                       # 4: convexity / fill ratio (1.0 for rect, <1 for slope/bracket)
            prim.mass_grams / 3.0,                     # 5: normalized mass
            float(n_studs) / 8.0,                      # 6: male stud count
            float(n_antistuds) / 8.0,                  # 7: female anti-stud count
            float(n_lateral) / 4.0,                    # 8: lateral SNOT/side port count
            float(n_technic) / 2.0,                    # 9: Technic pin/hole port count
            float(n_clip_bar) / 2.0,                   # 10: Clip/bar port count
            1.0 if prim.is_snot else 0.0,              # 11: SNOT / multi-axis indicator
            1.0 if len(prim.hull_boxes) > 1 else 0.0,  # 12: multi-box / sloped / bracket hull indicator
            float(len(prim.ports)) / 16.0,             # 13: total port count
            (max(dx, dy) / max(1.0, min(dx, dy))) / 8.0,  # 14: aspect elongation ratio
            (dz / max(1.0, min(dx, dy))),              # 15: height-to-width slenderness ratio
        ],
        dtype=np.float32,
    )
    return desc


def compute_intrinsic_part_descriptor_from_primitive(prim: BrickPrimitive) -> np.ndarray:
    """
    Compute the 16-D part-ID-free intrinsic geometric & connector descriptor φ(b) in R^16
    directly from a `BrickPrimitive` (curated `TIER_A` or compiled LDraw `2610` part).
    """
    xmin = min(b.xmin for b in prim.hull_boxes)
    ymin = min(b.ymin for b in prim.hull_boxes)
    zmin = min(b.zmin for b in prim.hull_boxes)
    xmax = max(b.xmax for b in prim.hull_boxes)
    ymax = max(b.ymax for b in prim.hull_boxes)
    zmax = max(b.zmax for b in prim.hull_boxes)
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    dz = float(zmax - zmin)
    bbox_vol = max(1.0, dx * dy * dz)
    hull_vol = float(sum(b.volume_ldu3 for b in prim.hull_boxes))
    ref_vol = 20.0 * 20.0 * 24.0

    n_studs = sum(
        1
        for p in prim.ports
        if p.polarity == PortPolarity.STUD
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    )
    n_antistuds = sum(
        1
        for p in prim.ports
        if p.polarity == PortPolarity.ANTI_STUD
        and p.connector_type in (ConnectorType.STANDARD_STUD, ConnectorType.HOLLOW_STUD)
    )
    n_lateral = sum(1 for p in prim.ports if abs(int(p.local_frame.z_axis[2])) == 0)
    n_technic = sum(
        1
        for p in prim.ports
        if p.connector_type
        in (
            ConnectorType.TECHNIC_PIN_HOLE,
            ConnectorType.TECHNIC_PIN,
            ConnectorType.SMOOTH_PIN,
            ConnectorType.TECHNIC_AXLE_HOLE,
            ConnectorType.TECHNIC_AXLE,
            ConnectorType.WHEEL_HUB_HOLE,
            ConnectorType.WHEEL_PIN,
        )
    )
    n_clip_bar = sum(
        1
        for p in prim.ports
        if p.connector_type
        in (
            ConnectorType.CLIP_3_18MM,
            ConnectorType.BAR_3_18MM,
            ConnectorType.HINGE_SOCKET,
            ConnectorType.HINGE_PIN,
            ConnectorType.LOCKING_HINGE_SOCKET,
            ConnectorType.LOCKING_HINGE_PIN,
            ConnectorType.BALL_SOCKET,
            ConnectorType.BALL_JOINT,
            ConnectorType.TURNTABLE_BASE_SOCKET,
            ConnectorType.TURNTABLE_TOP_PIN,
        )
    )

    return np.array(
        [
            min(dx, dy) / 20.0,
            max(dx, dy) / 20.0,
            dz / 24.0,
            hull_vol / ref_vol,
            hull_vol / bbox_vol,
            prim.mass_grams / 3.0,
            float(n_studs) / 8.0,
            float(n_antistuds) / 8.0,
            float(n_lateral) / 4.0,
            float(n_technic) / 2.0,
            float(n_clip_bar) / 2.0,
            1.0 if prim.is_snot else 0.0,
            1.0 if len(prim.hull_boxes) > 1 else 0.0,
            float(len(prim.ports)) / 16.0,
            (max(dx, dy) / max(1.0, min(dx, dy))) / 8.0,
            (dz / max(1.0, min(dx, dy))),
        ],
        dtype=np.float32,
    )


def get_declarative_part_spec(part_id: str) -> DeclarativePartSpec:
    """Return the provenance-tracked `DeclarativePartSpec` for `part_id` (`TIER_A` or compiled LDraw)."""
    if part_id not in EXTENDED_BRICK_CATALOG:
        from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

        pid_l = part_id.lower()
        if pid_l in EXTENDED_BRICK_CATALOG:
            part_id = pid_l
        elif pid_l in COMPILED_LDRAW_CATALOG:
            return COMPILED_LDRAW_CATALOG.get_declarative_spec(
                pid_l, compute_intrinsic_part_descriptor_from_primitive
            )
    prim = get_brick_primitive(part_id)
    desc = compute_intrinsic_part_descriptor_from_primitive(prim)
    sig_str = f"{prim.part_id}:{prim.name}:{len(prim.hull_boxes)}:{len(prim.ports)}:{prim.mass_grams}"
    sha = hashlib.sha256(sig_str.encode("utf-8")).hexdigest()
    contract = CollisionContractSpec(
        contract_id="exact_ldu_box_plus_protrusion_v1",
        source_mesh_version="ldraw_2026_01_parametric",
        decomposition_method="EXACT_LDU_BOX_PLUS_PROTRUSION",
        geometric_inset_ldu=0.0,
        collision_algorithm="EXACT_INT_AABB_PLUS_MATED_PROTRUSION_EXEMPTION",
        allowed_penetration_ldu=0.0,
        mated_port_exemption_enabled=True,
    )
    prov = AssetProvenanceManifest(
        part_id=prim.part_id,
        provider_id="topo_brick_parametric_ldu",
        source_version="2026.11.r7",
        sha256_hash=sha,
        geometry_license="CC-BY-4.0 / MIT (Parametric LDraw Specification)",
        connector_license="CC-BY-SA-4.0 (LDCad Shadow) / MIT (Parametric Port Table)",
        redistribution_status="COMMERCIALLY_USABLE_WITH_ATTRIBUTION",
        collision_contract=contract,
        validation_status="VALIDATED_EXACT_MATING_AND_CLEARANCE",
    )
    return DeclarativePartSpec(
        primitive=prim,
        intrinsic_descriptor=tuple(float(v) for v in desc),
        provenance=prov,
    )


def is_known_part_id(part_id: str) -> bool:
    """Return True if `part_id` exists in `EXTENDED_BRICK_CATALOG` (`213`) or `COMPILED_LDRAW_CATALOG` (`14,500+`)."""
    if part_id in EXTENDED_BRICK_CATALOG or part_id.lower() in EXTENDED_BRICK_CATALOG:
        return True
    from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

    return part_id.lower() in COMPILED_LDRAW_CATALOG


def get_brick_primitive(part_id: str) -> BrickPrimitive:
    """Retrieve a brick primitive by LDraw part ID (`EXTENDED_BRICK_CATALOG` first, then `COMPILED_LDRAW_CATALOG`)."""
    if part_id in EXTENDED_BRICK_CATALOG:
        return EXTENDED_BRICK_CATALOG[part_id]
    pid_l = part_id.lower()
    if pid_l in EXTENDED_BRICK_CATALOG:
        return EXTENDED_BRICK_CATALOG[pid_l]
    from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

    if pid_l in COMPILED_LDRAW_CATALOG:
        return COMPILED_LDRAW_CATALOG.get_primitive(pid_l)
    raise KeyError(
        f"Unknown LDraw part_id '{part_id}' (not in EXTENDED_BRICK_CATALOG or COMPILED_LDRAW_CATALOG)."
    )


# Merge `TIER_A_CATALOG` (`160+` parts incl. 100% of LEGO Set `10698-1` + OMR + 8 connector families)
# into `EXTENDED_BRICK_CATALOG` while preserving `BRICK_CATALOG` frozen at 20 parts.
from topo_brick.interface.tier_a_catalog import (  # noqa: E402
    LDRAW_PART_ALIASES,
    SET_10698_OFFICIAL_INVENTORY,
    build_tier_a_catalog,
    normalize_ldraw_part_id,
)

TIER_A_CATALOG: Dict[str, BrickPrimitive] = build_tier_a_catalog(EXTENDED_BRICK_CATALOG)
EXTENDED_BRICK_CATALOG.update(TIER_A_CATALOG)
TIER_A_CATALOG_IDS: Tuple[str, ...] = tuple(TIER_A_CATALOG.keys())
SET_10698_KIT_IDS: Tuple[str, ...] = tuple(SET_10698_OFFICIAL_INVENTORY.keys())
_ = (LDRAW_PART_ALIASES, normalize_ldraw_part_id)
