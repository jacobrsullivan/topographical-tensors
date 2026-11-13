"""
WP-L6a: LDraw (.ldr / .mpd) Parser, Exporter, and Canonical Topological Course Decomposer.

Provides:
1. `LDrawPartSpec`: Parsed representation of an LDraw type-1 line (`1 <color> <x> <y> <z> <R_3x3> <part>.dat`)
   mapped into exact `topo_brick` integer LDU coordinates (`Z^3_LDU ⋊ SO(3)_cubic`).
2. `parse_ldr_text` & `export_state_to_ldr_text`: Round-trip conversion between standard LDraw
   files (with `-Y` up or `+Z` up conventions) and `AssemblyState`.
3. `decompose_assembly_to_courses`: Peels a completed 3D brick assembly bottom-up (including
   sideways SNOT sub-attachments) into a canonical sequence of teacher course trajectories
   `(state_t, frontier_t, pool_t, teacher_candidate_indices)`, matching each ground-truth brick
   to its exact candidate index in `generate_candidate_pool`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.brick_catalog import BRICK_CATALOG, BrickPrimitive
from topo_brick.interface.ldu_frames import (
    LDUBox,
    LDUFrame,
    matrix_to_rot_id,
    rot_id_to_matrix,
)
from topo_brick.interface.port_manifold import (
    ActiveConnectionFrontier,
    AssemblyState,
    CandidatePool,
)
from topo_brick.representation.interface_fields import TargetShapeBoxes


@dataclass(frozen=True, slots=True)
class LDrawPartSpec:
    """A single brick placement parsed from or exported to an LDraw type-1 line."""

    part_id: str
    pose: LDUFrame
    color_code: int = 15
    is_grounded: bool = False


@dataclass(frozen=True, slots=True)
class LDrawParseReport:
    """
    WP-L8a: Explicit LDraw Integer-Lattice & 20-Part Subset Validation Report.
    Tracks accepted `LDrawPartSpec` placements alongside explicit counts and line numbers
    of rejected type-1 lines (unknown catalog parts, fractional off-lattice coordinates,
    or scaled/non-cubic rotation matrices outside SO(3)_cubic).
    """

    accepted_specs: Tuple[LDrawPartSpec, ...]
    total_type1_lines: int
    rejected_unknown_part_lines: Tuple[int, ...]
    rejected_off_lattice_coord_lines: Tuple[int, ...]
    rejected_non_cubic_rotation_lines: Tuple[int, ...]
    rejected_malformed_lines: Tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CourseTrajectoryStep:
    """
    One canonical teacher course step extracted from a completed 3D assembly.
    """

    step_index: int
    state_before: AssemblyState
    frontier: ActiveConnectionFrontier
    pool: CandidatePool
    teacher_candidate_indices: Tuple[int, ...]
    target_shape: TargetShapeBoxes


def parse_ldr_text_with_report(
    ldr_text: str,
    use_z_up: bool = True,
    strict: bool = False,
    catalog: Optional[Dict[str, BrickPrimitive]] = None,
    alias_map: Optional[Dict[str, str]] = None,
) -> LDrawParseReport:
    """
    WP-L8a: Parse LDraw `.ldr` text under the explicit integer-lattice & catalog contract:
      1. Part ID must belong to `catalog` (defaults to the 20-part `BRICK_CATALOG`).
      2. Position (x, y, z) must lie on the integer LDU lattice Z^3_LDU (within 1e-4 tolerance).
      3. Rotation 3x3 matrix must belong to the 24 proper cubic orientations SO(3)_cubic.
    Lines violating (1)-(3) are explicitly logged and rejected (or raise ValueError if `strict=True`)
    rather than being silently rounded or normalized.
    """
    active_catalog = catalog if catalog is not None else BRICK_CATALOG
    specs: List[LDrawPartSpec] = []
    total_type1 = 0
    rej_part: List[int] = []
    rej_coord: List[int] = []
    rej_rot: List[int] = []
    rej_malformed: List[int] = []

    for line_no, raw_line in enumerate(ldr_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or not line.startswith("1 "):
            continue
        tokens = line.split()
        if len(tokens) < 15:
            rej_malformed.append(line_no)
            if strict:
                raise ValueError(f"Line {line_no}: Malformed LDraw type-1 line '{line}'.")
            continue
        total_type1 += 1
        try:
            color_code = int(tokens[1])
        except ValueError as exc:
            rej_malformed.append(line_no)
            if strict:
                raise ValueError(f"Line {line_no}: Malformed color code '{tokens[1]}'.") from exc
            continue

        part_token = tokens[14].lower()
        if part_token.endswith(".dat"):
            part_token = part_token[:-4]
        if alias_map is not None and part_token in alias_map:
            part_token = alias_map[part_token]
        if part_token not in active_catalog:
            rej_part.append(line_no)
            if strict:
                raise ValueError(f"Line {line_no}: Part '{part_token}' not in active catalog.")
            continue

        # Verify integer LDU coordinates (reject fractional/off-lattice positions)
        coord_floats = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        if any(abs(v - round(v)) > 1e-4 for v in coord_floats):
            rej_coord.append(line_no)
            if strict:
                raise ValueError(f"Line {line_no}: Off-lattice fractional coordinates {coord_floats}.")
            continue
        x_raw, y_raw, z_raw = (int(round(v)) for v in coord_floats)

        # Verify proper cubic rotation matrix in SO(3)_cubic
        mat_floats = [float(t) for t in tokens[5:14]]
        if any(abs(v - round(v)) > 1e-4 for v in mat_floats):
            rej_rot.append(line_no)
            if strict:
                raise ValueError(f"Line {line_no}: Non-integer/scaled rotation matrix {mat_floats}.")
            continue
        mat_vals = [int(round(v)) for v in mat_floats]
        rot_mat = np.array(mat_vals, dtype=np.int64).reshape(3, 3)

        if not use_z_up:
            # Convert standard LDraw (-Y up) to topo_brick (+Z up): (x, z, -y)
            c_perm = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.int64)
            rot_mat = c_perm @ rot_mat @ c_perm.T
            pos = (x_raw, z_raw, -y_raw)
        else:
            pos = (x_raw, y_raw, z_raw)

        try:
            rot_id = matrix_to_rot_id(rot_mat)
        except ValueError as exc:
            rej_rot.append(line_no)
            if strict:
                raise ValueError(f"Line {line_no}: Matrix not in SO(3)_cubic: {exc}") from exc
            continue

        pose = LDUFrame(rot_id=rot_id, pos=pos)
        prim = active_catalog[part_token]
        w_boxes = [b.transform(pose) for b in prim.hull_boxes]
        min_z = min(b.zmin for b in w_boxes)
        specs.append(
            LDrawPartSpec(
                part_id=part_token,
                pose=pose,
                color_code=color_code,
                is_grounded=(min_z <= 0),
            )
        )

    return LDrawParseReport(
        accepted_specs=tuple(specs),
        total_type1_lines=total_type1,
        rejected_unknown_part_lines=tuple(rej_part),
        rejected_off_lattice_coord_lines=tuple(rej_coord),
        rejected_non_cubic_rotation_lines=tuple(rej_rot),
        rejected_malformed_lines=tuple(rej_malformed),
    )


def parse_ldr_text(
    ldr_text: str,
    use_z_up: bool = True,
    strict: bool = False,
    catalog: Optional[Dict[str, BrickPrimitive]] = None,
) -> List[LDrawPartSpec]:
    """
    Parse LDraw `.ldr` text containing type-1 part placement lines:
        `1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part>.dat`
    into exact `LDrawPartSpec` objects in `catalog` (defaults to `BRICK_CATALOG`).
    """
    report = parse_ldr_text_with_report(ldr_text, use_z_up=use_z_up, strict=strict, catalog=catalog)
    return list(report.accepted_specs)


def export_state_to_ldr_text(state: AssemblyState, model_name: str = "topo_brick_model.ldr") -> str:
    """
    Export an `AssemblyState` to canonical LDraw `.ldr` format (`+Z` up integer LDU convention).
    """
    lines: List[str] = [
        f"0 {model_name}",
        "0 Name: " + model_name,
        "0 Author: topo_brick",
    ]
    for inst_id, placed in sorted(state.placed_parts.items()):
        x, y, z = placed.pose.pos
        r_mat = rot_id_to_matrix(placed.pose.rot_id).reshape(-1)
        r_str = " ".join(str(int(v)) for v in r_mat)
        color = 14 if placed.is_grounded else (1 + (inst_id % 14))
        lines.append(f"1 {color} {x} {y} {z} {r_str} {placed.part_id}.dat")
    lines.append("0 NOFILE")
    return "\n".join(lines) + "\n"


def decompose_assembly_to_courses(
    part_specs: Sequence[LDrawPartSpec],
    allowed_part_ids: Optional[Sequence[str]] = None,
    max_bricks_per_course: int = 4,
    catalog: Optional[Dict[str, BrickPrimitive]] = None,
) -> List[CourseTrajectoryStep]:
    """
    Deterministically decompose a target 3D assembly `part_specs` into a bottom-up sequence
    of `CourseTrajectoryStep` teacher tuples `(state_t, frontier_t, pool_t, teacher_indices)`.
    Matches each unplaced ground-truth brick to its exact candidate in `pool_t` by
    canonical `(part_id, world_boxes, port_sig)`.
    """
    if not part_specs:
        return []

    active_catalog = catalog if catalog is not None else BRICK_CATALOG
    state = AssemblyState()
    remaining: List[Tuple[LDrawPartSpec, Tuple[LDUBox, ...], frozenset]] = []
    all_target_boxes: List[LDUBox] = []

    for spec in part_specs:
        prim = active_catalog[spec.part_id]
        w_boxes = tuple(b.transform(spec.pose) for b in prim.hull_boxes)
        port_sig = frozenset(
            (spec.pose.compose(p.local_frame).pos, int(p.polarity))
            for p in prim.ports
        )
        if spec.is_grounded:
            state.place_root_part(spec.part_id, spec.pose, is_grounded=True)
        else:
            remaining.append((spec, w_boxes, port_sig))
            all_target_boxes.extend(w_boxes)

    target_shape = TargetShapeBoxes(boxes=tuple(all_target_boxes))
    inferred_allowed = (
        list(allowed_part_ids)
        if allowed_part_ids is not None
        else sorted({s.part_id for s in part_specs})
    )
    validator = SparseStaticEquilibriumValidator()
    steps: List[CourseTrajectoryStep] = []
    step_idx = 0

    while remaining:
        frontier = state.get_active_frontier()
        pool = state.generate_candidate_pool(allowed_part_ids=inferred_allowed, frontier=frontier)

        # Map canonical (part_id, world_boxes, port_sig) -> candidate_id in pool
        sig_to_cid = {}
        for cid, cand in enumerate(pool.candidates):
            prim = active_catalog[cand.part_id]
            p_sig = frozenset(
                (cand.pose.compose(p.local_frame).pos, int(p.polarity))
                for p in prim.ports
            )
            sig_to_cid[(cand.part_id, cand.world_boxes, p_sig)] = cid

        # Find all remaining ground-truth bricks currently placeable on F_t
        attachable: List[Tuple[int, int, int]] = []  # (min_z, rem_idx, cid)
        for r_idx, (spec, w_boxes, p_sig) in enumerate(remaining):
            key = (spec.part_id, w_boxes, p_sig)
            if key in sig_to_cid:
                min_z = min(b.zmin for b in w_boxes)
                attachable.append((min_z, r_idx, sig_to_cid[key]))

        if not attachable:
            break

        # Group by lowest available Z course elevation first
        attachable.sort(key=lambda item: (item[0], item[1]))
        lowest_z = attachable[0][0]
        course_candidates = [item for item in attachable if item[0] <= lowest_z + 8]

        selected_cids: List[int] = []
        selected_rem_indices: Set[int] = set()
        blocked_cids: Set[int] = set()

        for _, r_idx, cid in course_candidates:
            if cid in blocked_cids:
                continue
            selected_cids.append(cid)
            selected_rem_indices.add(r_idx)
            blocked_cids.add(cid)
            blocked_cids.update(pool.conflict_adjacency[cid])
            if len(selected_cids) >= max_bricks_per_course:
                break

        import copy

        state_snapshot = copy.deepcopy(state)
        state.commit_course(pool, selected_cids)
        # Verify the committed teacher course maintains static equilibrium
        eq = validator.evaluate(state)
        if not eq.is_stable:
            raise RuntimeError(f"Decomposed teacher course at step {step_idx} is unstable: {eq}")

        steps.append(
            CourseTrajectoryStep(
                step_index=step_idx,
                state_before=state_snapshot,
                frontier=frontier,
                pool=pool,
                teacher_candidate_indices=tuple(selected_cids),
                target_shape=target_shape,
            )
        )
        remaining = [
            item for idx, item in enumerate(remaining) if idx not in selected_rem_indices
        ]
        step_idx += 1

    return steps
