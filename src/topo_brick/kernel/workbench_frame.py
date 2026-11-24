"""
World-to-Workbench Coordinate Frame Transform (`Phase 16 Decision 1`).

Preserves caller world coordinate frames `(ox, oy, oz)` across local workbench sub-voxel
synthesis (`open_catalog_sculpted` and `adaptive_plate_slope`), translating:
  - Forward (`world -> local` by `-origin_ldu`): `coarse_target_field`, `AdaptiveSurfaceTarget`,
    `forbidden_support_boxes`, `suppress_studless_capping_boxes`, `LandmarkAttachmentRequest`,
    `verts_ldu`, `symmetry_plane_coord_ldu`, and `ground_plane_z_ldu`.
  - Inverse (`local -> world` by `+origin_ldu`): `AssemblyState` (`PlacedPart` poses, `is_grounded`,
    and `recorded_step_ids`), `ExecutionTrace` (including recursive `subtrace`s),
    `evaluated_target_field`, and diagnostic `BilateralSymmetryReport` / `RegionalAppearanceReport`
    instance IDs and mirror coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
import math
from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.kernel.kernel_contracts import AtomicBuildStep, ExecutionTrace
from topo_brick.representation.interface_fields import SparseTiledTargetField, TargetShapeBoxes


class UnsupportedWorkbenchFrameError(ValueError):
    """Raised when a target or ground-plane datum cannot be represented on the discrete workbench."""


def validate_workbench_origin_and_datum(
    origin_ldu: Tuple[int, int, int],
    world_ground_plane_z_ldu: float,
    *,
    target_zmin_ldu: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Validate that `origin_ldu = (ox, oy, oz)` and `world_ground_plane_z_ldu` are representable
    on the `10 LDU` half-stud XY / `2 LDU` sub-plate Z lattice and that target geometry does
    not extend below the declared solid ground datum.
    """
    if not math.isfinite(float(world_ground_plane_z_ldu)):
        raise UnsupportedWorkbenchFrameError(
            f"ground_plane_z_ldu must be finite, got {world_ground_plane_z_ldu!r}"
        )
    gz_round = int(round(float(world_ground_plane_z_ldu)))
    if abs(float(world_ground_plane_z_ldu) - float(gz_round)) > 1e-6 or (gz_round % 2) != 0:
        raise UnsupportedWorkbenchFrameError(
            f"ground_plane_z_ldu={world_ground_plane_z_ldu!r} must be an even integer LDU (2-LDU Z lattice)"
        )
    if len(origin_ldu) != 3 or any(isinstance(c, bool) or not isinstance(c, int) for c in origin_ldu):
        raise UnsupportedWorkbenchFrameError(
            f"origin_ldu must be a 3-tuple of ints, got {origin_ldu!r}"
        )
    ox, oy, oz = int(origin_ldu[0]), int(origin_ldu[1]), int(origin_ldu[2])
    if (ox % 10) != 0 or (oy % 10) != 0 or (oz % 2) != 0:
        raise UnsupportedWorkbenchFrameError(
            f"origin_ldu={origin_ldu!r} must align to (10, 10, 2) LDU lattice"
        )
    if target_zmin_ldu is not None and int(target_zmin_ldu) < gz_round:
        raise UnsupportedWorkbenchFrameError(
            f"target_below_ground_plane_z_ldu:{int(target_zmin_ldu)}<{gz_round}"
        )
    if gz_round < oz:
        raise UnsupportedWorkbenchFrameError(
            f"ground_plane_z_ldu={gz_round} lies below adaptive_target.origin_ldu[2]={oz}"
        )
    return (ox, oy, oz)


def shift_box_ldu(box: LDUBox, delta: Tuple[int, int, int]) -> LDUBox:
    dx, dy, dz = delta
    if dx == 0 and dy == 0 and dz == 0:
        return box
    return LDUBox(
        box.xmin + dx,
        box.ymin + dy,
        box.zmin + dz,
        box.xmax + dx,
        box.ymax + dy,
        box.zmax + dz,
    )


def shift_boxes_ldu(boxes: Sequence[LDUBox], delta: Tuple[int, int, int]) -> Tuple[LDUBox, ...]:
    if delta == (0, 0, 0) or not boxes:
        return tuple(boxes)
    return tuple(shift_box_ldu(b, delta) for b in boxes)


def shift_target_field_ldu(
    target_field: SparseTiledTargetField | TargetShapeBoxes | Any,
    delta: Tuple[int, int, int],
) -> SparseTiledTargetField | TargetShapeBoxes | Any:
    if delta == (0, 0, 0) or target_field is None:
        return target_field
    t_boxes = getattr(target_field, "boxes", None)
    if t_boxes is None:
        return target_field
    shifted = shift_boxes_ldu(t_boxes, delta)
    if isinstance(target_field, TargetShapeBoxes):
        return TargetShapeBoxes(boxes=shifted)
    return SparseTiledTargetField(boxes=shifted)


def _shift_atomic_step(
    step: AtomicBuildStep,
    delta: Tuple[int, int, int],
    world_ground_plane_z_ldu: float,
    iid_map: Dict[int, int],
) -> AtomicBuildStep:
    dx, dy, dz = delta
    new_pose: Optional[LDUFrame] = step.pose
    if step.pose is not None and step.op_type.value != "PLACE_SUBASSEMBLY":
        new_pose = LDUFrame(
            step.pose.rot_id,
            (step.pose.pos[0] + dx, step.pose.pos[1] + dy, step.pose.pos[2] + dz),
        )
    elif step.pose is not None and step.op_type.value == "PLACE_SUBASSEMBLY":
        new_pose = LDUFrame(
            step.pose.rot_id,
            (step.pose.pos[0] + dx, step.pose.pos[1] + dy, step.pose.pos[2] + dz),
        )
    new_iids = tuple(iid_map.get(i, i) for i in step.instance_ids)
    # Note: inside a `PLACE_SUBASSEMBLY`, `subtrace` poses are local workbench coordinates (`delta=(0,0,0)`)
    sub_delta = (0, 0, 0) if step.op_type.value == "PLACE_SUBASSEMBLY" else delta
    new_sub = tuple(
        _shift_atomic_step(s, sub_delta, 0.0 if sub_delta == (0, 0, 0) else world_ground_plane_z_ldu, iid_map)
        for s in step.subtrace
    )
    return AtomicBuildStep(
        op_type=step.op_type,
        instance_ids=new_iids,
        part_id=step.part_id,
        pose=new_pose,
        is_grounded=step.is_grounded,
        temp_support_tag=step.temp_support_tag,
        subtrace=new_sub,
        gravity_unit=step.gravity_unit,
    )


@dataclass(frozen=True, slots=True)
class LocalWorkbenchInputs:
    """Inputs shifted by `-origin_ldu` for zero-origin local workbench synthesis."""

    origin_ldu: Tuple[int, int, int]
    world_ground_plane_z_ldu: float
    local_ground_plane_z_ldu: float
    local_coarse_target_field: Any
    local_adaptive_target: Any
    local_forbidden_support_boxes: Tuple[LDUBox, ...]
    local_suppress_capping_boxes: Tuple[LDUBox, ...]
    local_landmark_requests: Tuple[Any, ...]
    local_verts_ldu: Optional[np.ndarray]
    local_symmetry_plane_coord_ldu: Optional[float]


class WorkbenchFrameTransform:
    """Lossless bidirectional transform between world LDU and a zero-origin local workbench."""

    @staticmethod
    def to_local_workbench(
        *,
        coarse_target_field: Any,
        adaptive_target: Any,
        world_ground_plane_z_ldu: float,
        forbidden_support_boxes: Sequence[LDUBox] = (),
        suppress_studless_capping_boxes: Sequence[LDUBox] = (),
        landmark_requests: Sequence[Any] = (),
        verts_ldu: Optional[np.ndarray] = None,
        symmetry_plane_axis: Optional[str] = None,
        symmetry_plane_coord_ldu: Optional[float] = None,
    ) -> LocalWorkbenchInputs:
        origin = validate_workbench_origin_and_datum(
            getattr(adaptive_target, "origin_ldu", (0, 0, 0)),
            world_ground_plane_z_ldu,
        )
        ox, oy, oz = origin
        if origin == (0, 0, 0):
            return LocalWorkbenchInputs(
                origin_ldu=origin,
                world_ground_plane_z_ldu=float(world_ground_plane_z_ldu),
                local_ground_plane_z_ldu=float(world_ground_plane_z_ldu),
                local_coarse_target_field=coarse_target_field,
                local_adaptive_target=adaptive_target,
                local_forbidden_support_boxes=tuple(forbidden_support_boxes),
                local_suppress_capping_boxes=tuple(suppress_studless_capping_boxes),
                local_landmark_requests=tuple(landmark_requests),
                local_verts_ldu=verts_ldu,
                local_symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
            )

        neg = (-ox, -oy, -oz)
        loc_coarse = shift_target_field_ldu(coarse_target_field, neg)
        loc_surf_pts = (
            np.asarray(adaptive_target.mesh_surf_pts, dtype=np.float64)
            - np.array([[float(ox), float(oy), float(oz)]], dtype=np.float64)
            if adaptive_target.mesh_surf_pts.shape[0] > 0
            else adaptive_target.mesh_surf_pts
        )
        loc_ad = dc_replace(
            adaptive_target,
            mesh_surf_pts=loc_surf_pts,
            origin_ldu=(0, 0, 0),
        )
        loc_forb = shift_boxes_ldu(forbidden_support_boxes, neg)
        loc_supp = shift_boxes_ldu(suppress_studless_capping_boxes, neg)
        loc_lms = tuple(
            dc_replace(req, anchor_box_ldu=shift_box_ldu(req.anchor_box_ldu, neg))
            for req in landmark_requests
        )
        loc_verts: Optional[np.ndarray] = None
        if verts_ldu is not None:
            v_arr = np.asarray(verts_ldu, dtype=np.float64)
            loc_verts = (
                v_arr - np.array([[float(ox), float(oy), float(oz)]], dtype=np.float64)
                if v_arr.shape[0] > 0
                else v_arr
            )
        loc_sym_coord = symmetry_plane_coord_ldu
        if symmetry_plane_coord_ldu is not None and symmetry_plane_axis is not None:
            ax = str(symmetry_plane_axis).lower().strip()
            if ax == "x":
                loc_sym_coord = float(symmetry_plane_coord_ldu) - float(ox)
            elif ax == "y":
                loc_sym_coord = float(symmetry_plane_coord_ldu) - float(oy)

        return LocalWorkbenchInputs(
            origin_ldu=origin,
            world_ground_plane_z_ldu=float(world_ground_plane_z_ldu),
            local_ground_plane_z_ldu=float(world_ground_plane_z_ldu) - float(oz),
            local_coarse_target_field=loc_coarse,
            local_adaptive_target=loc_ad,
            local_forbidden_support_boxes=loc_forb,
            local_suppress_capping_boxes=loc_supp,
            local_landmark_requests=loc_lms,
            local_verts_ldu=loc_verts,
            local_symmetry_plane_coord_ldu=loc_sym_coord,
        )

    @staticmethod
    def to_world_frame(
        *,
        local_state: AssemblyState,
        local_fine_target_field: Any,
        local_diagnostics: Optional[Any],
        origin_ldu: Tuple[int, int, int],
        world_ground_plane_z_ldu: float,
    ) -> Tuple[AssemblyState, Any, Optional[Any]]:
        if origin_ldu == (0, 0, 0):
            return local_state, local_fine_target_field, local_diagnostics

        ox, oy, oz = origin_ldu
        world_state = AssemblyState()
        step_order = (
            list(local_state.recorded_step_ids)
            if local_state.has_execution_trace and local_state.recorded_step_ids
            else sorted(local_state.placed_parts.keys())
        )
        seen = set(step_order)
        for rem_id in sorted(local_state.placed_parts.keys()):
            if rem_id not in seen:
                step_order.append(rem_id)

        iid_map: Dict[int, int] = {}
        for old_iid in step_order:
            if old_iid not in local_state.placed_parts:
                continue
            pl = local_state.placed_parts[old_iid]
            w_pose = LDUFrame(
                pl.pose.rot_id,
                (pl.pose.pos[0] + ox, pl.pose.pos[1] + oy, pl.pose.pos[2] + oz),
            )
            w_min_z = float(min(b.zmin + oz for b in pl.world_boxes)) if pl.world_boxes else float(w_pose.pos[2])
            w_grounded = bool(abs(w_min_z - float(world_ground_plane_z_ldu)) <= 0.25)
            new_iid = world_state.place_root_part(
                pl.part_id,
                w_pose,
                is_grounded=w_grounded,
                color_id=pl.color_id,
            )
            iid_map[old_iid] = new_iid

        if not local_state.has_execution_trace:
            world_state.mark_untraced()

        loc_trace: Optional[ExecutionTrace] = getattr(local_state, "_execution_trace", None)
        if loc_trace is not None:
            w_steps = tuple(
                _shift_atomic_step(st, origin_ldu, world_ground_plane_z_ldu, iid_map)
                for st in loc_trace.steps
            )
            setattr(
                world_state,
                "_execution_trace",
                ExecutionTrace(
                    steps=w_steps,
                    allow_subassemblies=loc_trace.allow_subassemblies,
                    allow_temp_supports=loc_trace.allow_temp_supports,
                ),
            )

        world_fine_tf = shift_target_field_ldu(local_fine_target_field, origin_ldu)

        world_diag = local_diagnostics
        if local_diagnostics is not None:
            ra = getattr(local_diagnostics, "regional_appearance_report", None)
            if ra is not None and getattr(ra, "high_gradient_instance_ids", ()):
                ra = dc_replace(
                    ra,
                    high_gradient_instance_ids=tuple(
                        iid_map.get(i, i) for i in ra.high_gradient_instance_ids
                    ),
                )
            sym = getattr(local_diagnostics, "bilateral_symmetry_report", None)
            if sym is not None:
                shift_c = float(ox) if sym.symmetry_axis == "x" else float(oy)
                sym = dc_replace(sym, plane_coord_ldu=round(float(sym.plane_coord_ldu) + shift_c, 6))
            world_diag = dc_replace(
                local_diagnostics,
                regional_appearance_report=ra,
                bilateral_symmetry_report=sym,
            )

        return world_state, world_fine_tf, world_diag
