"""
Phase 11 (`Ext 4A` & `Ext 4B`): Compositional Multi-Submodel (`.mpd`) Scene-Graph Realizer
(`SubmodelNodeSpec`, `realize_compositional_mpd`, `export_compositional_mpd`) and Semantic
Landmark Port Attacher (`LandmarkAttachmentRequest`, `attach_semantic_landmarks`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.independent_verifier import check_fine_csg_voxel_collision_2ldu
from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    ColoredInventorySpec,
    ConnectorType,
)
from topo_brick.interface.landmark_detail_index import (
    LANDMARK_DETAIL_INDEX,
    LandmarkMoldRecord,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState, _get_anchor_templates
from topo_brick.interface.se3_kinematics import (
    ContinuousSE3Pose,
    _C_LDRAW_TO_ZUP,
    _topo_brick_se3_to_ldraw_line,
)
from topo_brick.kernel.adaptive_surface_and_macro_planner import (
    AdaptiveSurfaceTarget,
    HollowShellPolicy,
    apply_hollow_shell_policy,
)
from topo_brick.kernel.view_priority import ViewPrioritySpec
from topo_brick.representation.interface_fields import SparseTiledTargetField


@dataclass(frozen=True, slots=True)
class SubmodelNodeSpec:
    """Single modular `.mpd` subassembly node in a compositional architectural scene DAG (`Ext 4A`)."""

    submodel_name: str
    adaptive_target: AdaptiveSurfaceTarget
    world_offset_ldu: Tuple[int, int, int] = (0, 0, 0)
    parent_submodel: Optional[str] = None
    forbidden_support_boxes: Tuple[LDUBox, ...] = ()
    hollow_shell_policy: Optional[HollowShellPolicy] = None
    semantic_style_hint: Optional[str] = None
    vertex_colors_rgb: Optional[np.ndarray] = None
    verts_ldu: Optional[np.ndarray] = None
    world_rot_id: int = 0


@dataclass(frozen=True, slots=True)
class LandmarkAttachmentRequest:
    """Declarative request to attach non-flush architectural/functional landmarks onto exposed ports (`Ext 4B`)."""

    role: str
    anchor_box_ldu: LDUBox
    preferred_color_id: Optional[int] = None
    count: int = 1
    license_filter: Optional[str] = None  # Inherits realize()'s license_filter when None


@dataclass(frozen=True, slots=True)
class SubmodelRealizationRecord:
    """Realized local subassembly state and world-instance mapping for `.mpd` export."""

    submodel_name: str
    world_offset_ldu: Tuple[int, int, int]
    local_part_entries: Tuple[Tuple[int, str, LDUFrame, int], ...]  # (world_iid, part_id, local_pose, color_id)
    world_rot_id: int = 0
    parent_submodel: Optional[str] = None


@dataclass(frozen=True, slots=True)
class CompositionalMPDResult:
    """Combined `AssemblyState`, recursive `ExecutionTrace`, target field, and multi-block `.mpd` text."""

    assembly: AssemblyState
    execution_trace: Any
    combined_target_field: SparseTiledTargetField
    submodel_records: Tuple[SubmodelRealizationRecord, ...]
    mpd_text: str
    landmarks_attached: int


def _toposort_submodels(nodes: Sequence[SubmodelNodeSpec]) -> List[SubmodelNodeSpec]:
    by_name = {n.submodel_name: n for n in nodes}
    visited: Set[str] = set()
    temp: Set[str] = set()
    ordered: List[SubmodelNodeSpec] = []

    def _dfs(name: str) -> None:
        if name in visited:
            return
        if name in temp:
            raise ValueError(f"Cycle detected in SubmodelNodeSpec DAG at {name!r}.")
        temp.add(name)
        node = by_name[name]
        if node.parent_submodel is not None:
            if node.parent_submodel not in by_name:
                raise ValueError(
                    f"Submodel {name!r} references unknown parent_submodel={node.parent_submodel!r}."
                )
            _dfs(node.parent_submodel)
        temp.remove(name)
        visited.add(name)
        ordered.append(node)

    for n in nodes:
        _dfs(n.submodel_name)
    return ordered


def _part_dict_for_csg(iid: int, part_id: str, pose: LDUFrame, is_grounded: bool) -> Dict[str, Any]:
    return {
        "instance_id": int(iid),
        "part_id": str(part_id),
        "rot_id": int(pose.rot_id),
        "pos": [int(c) for c in pose.pos],
        "is_grounded": bool(is_grounded),
    }


def attach_semantic_landmarks(
    kernel: Any,
    state: AssemblyState,
    requests: Sequence[LandmarkAttachmentRequest],
    default_license_filter: str = "all",
    allowed_part_ids: Optional[Sequence[str]] = None,
    rem_inv: Optional[Dict[str, int]] = None,
    foundation_policy: FoundationPolicy = FoundationPolicy.CLUTCHED_BASEPLATE,
    ground_plane_z_ldu: float = 0.0,
) -> List[Tuple[int, str, LDUFrame, bool]]:
    """
    Attach non-flush architectural & functional landmarks (`Phase 11 Extension 4B`) onto exposed
    frontier ports inside each `LandmarkAttachmentRequest.anchor_box_ldu`, strictly verifying:
      1. `license_filter` (`"cc_by_4_0_only"` emits `0` `CC-BY-SA-4.0` parts),
      2. Finite `rem_inv` non-overrun,
      3. `AssemblyState` hull/protrusion clearance + `2 LDU` CSG cavity clearance against neighbors,
      4. `check_monotone_insertion_corridor`, and
      5. `SparseStaticEquilibriumValidator` 6-DoF stability under `foundation_policy`.

    Returns `[(world_iid, part_id, world_pose, is_grounded), ...]` in exact attachment order.
    """
    from topo_brick.kernel.construction_kernel import check_monotone_insertion_corridor

    committed: List[Tuple[int, str, LDUFrame, bool]] = []
    if not requests or state.num_parts == 0:
        return committed

    for req in requests:
        if req.count <= 0:
            continue
        eff_lic = req.license_filter if req.license_filter is not None else default_license_filter
        role_records: Tuple[LandmarkMoldRecord, ...] = LANDMARK_DETAIL_INDEX.query_role(
            req.role, license_filter=eff_lic, allowed_part_ids=allowed_part_ids
        )
        if not role_records:
            continue

        abox = LDUBox(
            req.anchor_box_ldu.xmin - 12,
            req.anchor_box_ldu.ymin - 12,
            req.anchor_box_ldu.zmin - 12,
            req.anchor_box_ldu.xmax + 12,
            req.anchor_box_ldu.ymax + 12,
            req.anchor_box_ldu.zmax + 12,
        )

        cx_anchor = 0.5 * float(req.anchor_box_ldu.xmin + req.anchor_box_ldu.xmax)
        cy_anchor = 0.5 * float(req.anchor_box_ldu.ymin + req.anchor_box_ldu.ymax)

        placed_for_req = 0
        for _attempt in range(req.count):
            frontier = state.get_active_frontier()
            # Candidate exposed ports inside `abox`, preferring highest Z first (spires/roofs/parapets)
            cand_ports = [
                ep
                for ep in frontier.ports
                if ep.status.is_exposed
                and abox.xmin <= ep.world_frame.pos[0] <= abox.xmax
                and abox.ymin <= ep.world_frame.pos[1] <= abox.ymax
                and abox.zmin <= ep.world_frame.pos[2] <= abox.zmax
            ]
            # Prefer non-stud specialty ports (`BAR_3_18MM`, `HINGE_SOCKET`, `CLIP_3_18MM`) when matching role,
            # and tie-break equal-Z ports by proximity to `req.anchor_box_ldu` XY center (`TB-FIX-04`)!
            def _port_sort_key(ep: Any) -> Tuple[int, int, float, float, float, int, int]:
                is_spec = (
                    0
                    if (
                        (req.role == "CLIP_FLAG_BANNER" and ep.connector_type == ConnectorType.BAR_3_18MM)
                        or (
                            req.role == "HINGED_PANEL_MOUNT"
                            and ep.connector_type
                            in (
                                ConnectorType.HINGE_SOCKET,
                                ConnectorType.HINGE_PIN,
                                ConnectorType.LOCKING_HINGE_SOCKET,
                                ConnectorType.LOCKING_HINGE_PIN,
                            )
                        )
                    )
                    else 1
                )
                px_e = float(ep.world_frame.pos[0])
                py_e = float(ep.world_frame.pos[1])
                dx_a = abs(px_e - cx_anchor)
                dy_a = abs(py_e - cy_anchor)
                d2_xy = round(dx_a * dx_a + dy_a * dy_a, 4)
                return (
                    is_spec,
                    -int(ep.world_frame.pos[2]),
                    d2_xy,
                    round(dx_a, 4),
                    round(dy_a, 4),
                    int(ep.world_frame.pos[0]),
                    int(ep.world_frame.pos[1]),
                )

            cand_ports.sort(key=_port_sort_key)
            attached_one = False

            for ep in cand_ports:
                ux, uy, uz = ep.world_frame.pos
                ep_is_upward = ep.world_frame.z_axis == (0, 0, 1)
                for rec in role_records:
                    pid = rec.part_id
                    if rem_inv is not None and rem_inv.get(pid, 0) <= 0:
                        continue
                    tmpls = _get_anchor_templates(
                        ep.world_frame.rot_id, ep.polarity, ep.connector_type, pid
                    )
                    for tmpl in tmpls:
                        rid_c, dx_c, dy_c, dz_c, rboxes_c, _, _, _ = tmpl
                        # Keep architectural spires, domes, windows, doors, masonry, and upward-mounted
                        # antennas/masts (`BAR_POLE_ANTENNA`, `TB-FIX-04`) upright (`+Z` up)
                        pose_c = LDUFrame(rid_c, (ux + dx_c, uy + dy_c, uz + dz_c))
                        if (
                            req.role
                            in (
                                "FINIAL_SPIRE_CONE",
                                "DOME_ROOF_CAP",
                                "WINDOW_FRAME_LATTICE",
                                "DOOR_PORTCULLIS",
                                "TEXTURED_MASONRY_BRICK",
                            )
                            or (req.role == "BAR_POLE_ANTENNA" and ep_is_upward)
                        ) and pose_c.z_axis != (0, 0, 1):
                            continue
                        wbs_c = tuple(
                            LDUBox(b[0] + ux, b[1] + uy, b[2] + uz, b[3] + ux, b[4] + uy, b[5] + uz)
                            for b in rboxes_c
                        )
                        min_wb_z = float(min(b.zmin for b in wbs_c))
                        if min_wb_z < float(ground_plane_z_ldu) - 0.25:
                            continue
                        if state.check_boxes_collision(wbs_c):
                            continue
                        if not check_monotone_insertion_corridor(state, pid, pose_c):
                            continue

                        tok = None
                        try:
                            tok = state.push_trial_part(
                                pid,
                                pose_c,
                                is_grounded=(abs(min_wb_z - float(ground_plane_z_ldu)) <= 0.25),
                            )
                            if len(tok.added_adj_other_insts) == 0:
                                state.pop_trial_part(tok)
                                continue
                            # Check `2 LDU` CSG against nearby placed parts (`+/- 24 LDU` AABB neighborhood)
                            exp_b = LDUBox(
                                min(b.xmin for b in wbs_c) - 24,
                                min(b.ymin for b in wbs_c) - 24,
                                min(b.zmin for b in wbs_c) - 24,
                                max(b.xmax for b in wbs_c) + 24,
                                max(b.ymax for b in wbs_c) + 24,
                                max(b.zmax for b in wbs_c) + 24,
                            )
                            local_csg_parts = [
                                _part_dict_for_csg(
                                    nb_id,
                                    state.placed_parts[nb_id].part_id,
                                    state.placed_parts[nb_id].pose,
                                    state.placed_parts[nb_id].is_grounded,
                                )
                                for nb_id in sorted(state.placed_parts.keys())
                                if any(nb_b.intersects(exp_b) for nb_b in state.placed_parts[nb_id].world_boxes)
                            ]
                            csg_ok, _ = check_fine_csg_voxel_collision_2ldu(local_csg_parts)
                            if not csg_ok:
                                state.pop_trial_part(tok)
                                continue
                            eq_ok = kernel.validator.evaluate(state, foundation_policy=foundation_policy).is_stable
                            state.pop_trial_part(tok)
                            if not eq_ok:
                                continue
                        except ValueError:
                            if tok is not None:
                                state.pop_trial_part(tok)
                            continue

                        is_g = abs(min_wb_z - float(ground_plane_z_ldu)) <= 0.25
                        nid = state.place_root_part(
                            pid,
                            pose_c,
                            is_grounded=is_g,
                            color_id=int(req.preferred_color_id) if req.preferred_color_id is not None else 15,
                        )
                        if rem_inv is not None:
                            rem_inv[pid] = max(0, rem_inv.get(pid, 0) - 1)
                        committed.append((nid, pid, pose_c, is_g))
                        placed_for_req += 1
                        attached_one = True
                        break
                    if attached_one:
                        break
                if attached_one:
                    break
            if not attached_one:
                break
    return committed


def export_compositional_mpd(
    submodel_records: Sequence[SubmodelRealizationRecord],
    state: AssemblyState,
    extra_root_instance_ids: Sequence[int] = (),
    main_model_name: str = "main.ldr",
) -> str:
    """
    Export a true multi-block LDraw `.mpd` document (`0 FILE main.ldr` + nested Type-1 submodel
    references supporting non-identity `world_rot_id` & `parent_submodel` hierarchies, `TB-FIX-01`
    + `0 FILE <submodel_name> ... 0 STEP ... 0 NOFILE` blocks) guaranteed to round-trip `100%`
    through `flatten_omr_mpd_to_se3(..., catalog=COMBINED_BRICK_CATALOG)`.
    """
    rec_by_name: Dict[str, SubmodelRealizationRecord] = {
        rec.submodel_name: rec for rec in submodel_records
    }
    world_pose_by_name: Dict[str, ContinuousSE3Pose] = {
        rec.submodel_name: ContinuousSE3Pose.from_ldu_frame(
            LDUFrame(int(rec.world_rot_id), tuple(int(c) for c in rec.world_offset_ldu))
        )
        for rec in submodel_records
    }
    nested_children: Dict[str, List[SubmodelRealizationRecord]] = {}
    root_records: List[SubmodelRealizationRecord] = []
    for rec in submodel_records:
        if rec.parent_submodel is not None and rec.parent_submodel in rec_by_name:
            nested_children.setdefault(rec.parent_submodel, []).append(rec)
        else:
            root_records.append(rec)

    def _format_submodel_type1_line(pose_se3: ContinuousSE3Pose, ref_name: str) -> str:
        r_ld = _C_LDRAW_TO_ZUP.T @ pose_se3.R @ _C_LDRAW_TO_ZUP
        t_ld = _C_LDRAW_TO_ZUP.T @ pose_se3.t
        r_str = " ".join(f"{float(v):.6g}" for v in r_ld.reshape(-1))
        return f"1 16 {t_ld[0]:.6g} {t_ld[1]:.6g} {t_ld[2]:.6g} {r_str} {ref_name}"

    lines: List[str] = [
        f"0 FILE {main_model_name}",
        "0 Name: Topo-Brick Compositional MPD Assembly",
        "0 !LDRAW_ORG Model",
    ]
    for rec in root_records:
        lines.append(_format_submodel_type1_line(world_pose_by_name[rec.submodel_name], rec.submodel_name))

    for iid in extra_root_instance_ids:
        if iid in state.placed_parts:
            p = state.placed_parts[iid]
            lines.append(
                _topo_brick_se3_to_ldraw_line(
                    p.part_id, ContinuousSE3Pose.from_ldu_frame(p.pose), color_code=int(p.color_id)
                )
            )
    lines.append("0 STEP")
    lines.append("0 NOFILE")

    for rec in submodel_records:
        lines.append(f"0 FILE {rec.submodel_name}")
        lines.append(f"0 Name: {rec.submodel_name}")
        lines.append("0 !LDRAW_ORG Submodel")
        for w_iid, pid, loc_pose, fallback_c in rec.local_part_entries:
            eff_color = int(state.placed_parts[w_iid].color_id) if w_iid in state.placed_parts else int(fallback_c)
            lines.append(
                _topo_brick_se3_to_ldraw_line(
                    pid, ContinuousSE3Pose.from_ldu_frame(loc_pose), color_code=eff_color
                )
            )
            lines.append("0 STEP")
        parent_pose = world_pose_by_name[rec.submodel_name]
        for child_rec in nested_children.get(rec.submodel_name, ()):
            rel_pose = parent_pose.inverse().compose(world_pose_by_name[child_rec.submodel_name])
            lines.append(_format_submodel_type1_line(rel_pose, child_rec.submodel_name))
            lines.append("0 STEP")
        lines.append("0 NOFILE")

    return "\n".join(lines) + "\n"


def realize_compositional_mpd(
    kernel: Any,
    subassembly_graph: Sequence[SubmodelNodeSpec],
    allowed_part_ids: Optional[Sequence[str]] = None,
    part_retrieval: str = "open_catalog_top_k",
    retrieval_top_k: int = 32,
    license_filter: str = "all",
    semantic_style_hint: Optional[str] = None,
    inventory_counts: Optional[Dict[str, int]] = None,
    colored_inventory: Optional[ColoredInventorySpec] = None,
    landmark_requests: Sequence[LandmarkAttachmentRequest] = (),
    view_priority: Optional[ViewPrioritySpec] = None,
    support_policy: str = "minimal_permanent_pillars",
    foundation_policy: FoundationPolicy = FoundationPolicy.CLUTCHED_BASEPLATE,
) -> CompositionalMPDResult:
    """
    Realize a multi-submodel architectural scene DAG (`Phase 11 Extension 4A & 4B`):
      1. Traverses `subassembly_graph` in topological parent-before-child order sharing a single
         mutable `rem_inv`.
      2. Suppresses `"smooth_studless"` tile capping inside child attachment footprints on parents
         so child towers/gatehouses always find exposed upward studs to clutch onto!
      3. Verifies each child subassembly's local flat-table `ExecutionTrace` (`subtrace`), checks
         `check_rigid_group_insertion_corridor` into the merged assembly, mates interface ports,
         and emits recursive `AtomicOpType.PLACE_SUBASSEMBLY` steps.
      4. Attaches requested `landmark_requests` (`LandmarkDetailIndex`) and runs global two-zone
         chromatic allocation across the unified assembly if `colored_inventory` is provided.
    """
    from topo_brick.kernel.colored_inventory_allocator import (
        allocate_inventory_colors_min_cost,
        apply_chromatic_same_footprint_splits,
    )
    from topo_brick.kernel.construction_kernel import (
        AtomicBuildStep,
        AtomicOpType,
        ExecutionTrace,
        check_monotone_insertion_corridor,
        check_rigid_group_insertion_corridor,
    )
    from topo_brick.kernel.open_catalog_realizer import realize_open_catalog_sculpted

    ordered_nodes = _toposort_submodels(subassembly_graph)
    children_by_parent: Dict[str, List[SubmodelNodeSpec]] = {}
    for n in ordered_nodes:
        if n.parent_submodel is not None:
            children_by_parent.setdefault(n.parent_submodel, []).append(n)

    rem_inv: Optional[Dict[str, int]] = None
    if colored_inventory is not None:
        rem_inv = colored_inventory.marginal_mold_counts()
    elif inventory_counts is not None:
        rem_inv = dict(inventory_counts)

    eff_allowed_ids = allowed_part_ids
    if eff_allowed_ids is None and rem_inv is not None:
        eff_allowed_ids = tuple(sorted(pid for pid, q in rem_inv.items() if int(q) > 0))

    merged_state = AssemblyState()
    trace_steps: List[AtomicBuildStep] = []
    sub_records: List[SubmodelRealizationRecord] = []
    combined_boxes: List[LDUBox] = []

    for node in ordered_nodes:
        ox, oy, oz = (int(node.world_offset_ldu[0]), int(node.world_offset_ldu[1]), int(node.world_offset_ldu[2]))
        node_rot_id = int(node.world_rot_id)
        node_world_frame = LDUFrame(node_rot_id, (ox, oy, oz))
        node_inv_frame = node_world_frame.inverse()

        # Apply optional HollowShellPolicy on node's AdaptiveSurfaceTarget if not already applied
        ad_node = node.adaptive_target
        if (
            node.hollow_shell_policy is not None
            and node.hollow_shell_policy.enabled
            and ad_node.solid_occ_5x5x6 is None
        ):
            o2_h, o6_h = apply_hollow_shell_policy(
                ad_node.occ_5x5x2,
                ad_node.occ_5x5x6,
                node.hollow_shell_policy,
                forbidden_support_boxes=node.forbidden_support_boxes,
            )
            ad_node = AdaptiveSurfaceTarget(
                nx_studs=ad_node.nx_studs,
                ny_studs=ad_node.ny_studs,
                nz_courses=ad_node.nz_courses,
                occ_5x5x2=o2_h,
                occ_5x5x6=o6_h,
                mesh_surf_pts=ad_node.mesh_surf_pts,
                rasterize_ms=ad_node.rasterize_ms,
                solid_occ_5x5x6=ad_node.occ_5x5x6.copy(),
                hollow_shell_policy=node.hollow_shell_policy,
            )

        # Compute child attachment footprints in `node`'s local coordinates to suppress smooth tiles!
        suppress_boxes_local: List[LDUBox] = []
        for ch in children_by_parent.get(node.submodel_name, ()):
            ch_frame = LDUFrame(int(ch.world_rot_id), tuple(int(c) for c in ch.world_offset_ldu))
            ch_box_local = LDUBox(
                -4,
                -4,
                -16,
                ch.adaptive_target.nx_studs * 20 + 4,
                ch.adaptive_target.ny_studs * 20 + 4,
                16,
            )
            ch_box_in_parent = ch_box_local.transform(node_inv_frame.compose(ch_frame))
            suppress_boxes_local.append(ch_box_in_parent)

        eff_style = node.semantic_style_hint if node.semantic_style_hint is not None else semantic_style_hint
        # Realize subassembly in its local frame (`z_local >= 0`)
        sub_state, sub_fine_tf, _ = realize_open_catalog_sculpted(
            kernel=kernel,
            coarse_target_field=ad_node.to_fine_target_field(),
            adaptive_target=ad_node,
            allowed_part_ids=eff_allowed_ids,
            part_retrieval=part_retrieval,
            retrieval_top_k=retrieval_top_k,
            license_filter=license_filter,
            semantic_style_hint=eff_style,
            vertex_colors_rgb=node.vertex_colors_rgb if colored_inventory is None else None,
            verts_ldu=node.verts_ldu,
            inventory_counts=rem_inv,
            support_policy=support_policy,
            foundation_policy=FoundationPolicy.CLUTCHED_BASEPLATE if node.parent_submodel is not None else foundation_policy,
            forbidden_support_boxes=node.forbidden_support_boxes,
            suppress_studless_capping_boxes=tuple(suppress_boxes_local),
            view_priority=view_priority,
        )

        # Decrement shared `rem_inv` for parts consumed by `sub_state`
        # Note: `realize_open_catalog_sculpted` copies `inventory_counts`, so update `rem_inv` explicitly here!
        if rem_inv is not None:
            for p_sub in sub_state.placed_parts.values():
                rem_inv[p_sub.part_id] = max(0, rem_inv.get(p_sub.part_id, 0) - 1)

        # Transform `sub_fine_tf` boxes into world space
        for b_tf in sub_fine_tf.boxes:
            combined_boxes.append(b_tf.transform(node_world_frame))

        # Build local step order (preserving `sub_state.recorded_step_ids`)
        step_order_local = (
            list(sub_state.recorded_step_ids)
            if sub_state.recorded_step_ids
            else sorted(sub_state.placed_parts.keys())
        )

        if node.parent_submodel is None and oz == 0:
            # Root grounded submodel on the table (`z == 0`): emit atomic `PLACE_PART` steps on the
            # baseplate while keeping its own modular `0 FILE <submodel_name>` block in `.mpd`!
            loc_entries: List[Tuple[int, str, LDUFrame, int]] = []
            for lid in step_order_local:
                pl_l = sub_state.placed_parts[lid]
                w_pose = node_world_frame.compose(pl_l.pose)
                w_iid = merged_state.place_root_part(
                    pl_l.part_id, w_pose, is_grounded=pl_l.is_grounded, color_id=pl_l.color_id
                )
                loc_entries.append((w_iid, pl_l.part_id, pl_l.pose, pl_l.color_id))
                trace_steps.append(
                    AtomicBuildStep(
                        op_type=AtomicOpType.PLACE_PART,
                        instance_ids=(w_iid,),
                        part_id=pl_l.part_id,
                        pose=w_pose,
                        is_grounded=pl_l.is_grounded,
                    )
                )
            sub_records.append(
                SubmodelRealizationRecord(
                    submodel_name=node.submodel_name,
                    world_offset_ldu=(ox, oy, oz),
                    local_part_entries=tuple(loc_entries),
                    world_rot_id=node_rot_id,
                    parent_submodel=node.parent_submodel,
                )
            )
            continue

        # Multi-submodel compositional node:
        # Partition `sub_state` into physically connected components (ordered bottom-up) so every
        # `PLACE_SUBASSEMBLY` is a single rigid connected component on the workbench, while keeping
        # all parts grouped under `node.submodel_name` in the `.mpd` file!
        comp_sets: List[Set[int]] = []
        seen_lids: Set[int] = set()
        for lid_seed in step_order_local:
            if lid_seed in seen_lids:
                continue
            c_vis: Set[int] = {lid_seed}
            q_c = [lid_seed]
            while q_c:
                cur_l = q_c.pop()
                for nb_l in sub_state._part_adj.get(cur_l, ()):
                    if nb_l not in c_vis:
                        c_vis.add(nb_l)
                        q_c.append(nb_l)
            seen_lids |= c_vis
            comp_sets.append(c_vis)
        comp_sets.sort(
            key=lambda cs: (
                min(min(b.zmin for b in sub_state.placed_parts[i].world_boxes) for i in cs),
                min(cs),
            )
        )

        loc_entries_m: List[Tuple[int, str, LDUFrame, int]] = []
        for c_idx_sub, c_set in enumerate(comp_sets):
            comp_order = [lid for lid in step_order_local if lid in c_set]
            comp_min_z = min(min(b.zmin for b in sub_state.placed_parts[i].world_boxes) for i in comp_order)
            sim_local = AssemblyState()
            local_subtrace_steps: List[AtomicBuildStep] = []
            world_grp_boxes: List[LDUBox] = []
            for lid in comp_order:
                pl_l = sub_state.placed_parts[lid]
                if not check_monotone_insertion_corridor(sim_local, pl_l.part_id, pl_l.pose):
                    raise RuntimeError(f"Submodel {node.submodel_name!r} failed local monotone insertion at part {lid}.")
                is_loc_gr = bool(pl_l.is_grounded or min(b.zmin for b in pl_l.world_boxes) <= comp_min_z)
                sim_local.place_root_part(pl_l.part_id, pl_l.pose, is_grounded=is_loc_gr)
                for b_l in pl_l.world_boxes:
                    world_grp_boxes.append(b_l.transform(node_world_frame))

            if merged_state.num_parts > 0:
                if not check_rigid_group_insertion_corridor(merged_state, tuple(world_grp_boxes)):
                    raise RuntimeError(
                        f"Child submodel {node.submodel_name!r} at {(ox, oy, oz)} lacks a clear rigid-group insertion corridor!"
                    )

            comp_world_iids: List[int] = []
            for lid in comp_order:
                pl_l = sub_state.placed_parts[lid]
                w_pose = node_world_frame.compose(pl_l.pose)
                w_grounded = bool(oz == 0 and pl_l.is_grounded)
                w_iid = merged_state.place_root_part(
                    pl_l.part_id, w_pose, is_grounded=w_grounded, color_id=pl_l.color_id
                )
                comp_world_iids.append(w_iid)
                loc_entries_m.append((w_iid, pl_l.part_id, pl_l.pose, pl_l.color_id))
                is_loc_gr = bool(pl_l.is_grounded or min(b.zmin for b in pl_l.world_boxes) <= comp_min_z)
                local_subtrace_steps.append(
                    AtomicBuildStep(
                        op_type=AtomicOpType.PLACE_PART,
                        instance_ids=(w_iid,),
                        part_id=pl_l.part_id,
                        pose=pl_l.pose,
                        is_grounded=is_loc_gr,
                    )
                )

            if len(comp_world_iids) == 1:
                p1 = merged_state.placed_parts[comp_world_iids[0]]
                trace_steps.append(
                    AtomicBuildStep(
                        op_type=AtomicOpType.PLACE_PART,
                        instance_ids=(comp_world_iids[0],),
                        part_id=p1.part_id,
                        pose=p1.pose,
                        is_grounded=p1.is_grounded,
                    )
                )
            elif len(comp_world_iids) >= 2:
                sub_tag = node.submodel_name if len(comp_sets) == 1 else f"{node.submodel_name}_c{c_idx_sub}"
                trace_steps.append(
                    AtomicBuildStep(
                        op_type=AtomicOpType.PLACE_SUBASSEMBLY,
                        instance_ids=tuple(comp_world_iids),
                        part_id=sub_tag,
                        pose=node_world_frame,
                        is_grounded=(oz == 0),
                        subtrace=tuple(local_subtrace_steps),
                    )
                )
        sub_records.append(
            SubmodelRealizationRecord(
                submodel_name=node.submodel_name,
                world_offset_ldu=(ox, oy, oz),
                local_part_entries=tuple(loc_entries_m),
                world_rot_id=node_rot_id,
                parent_submodel=node.parent_submodel,
            )
        )

    # 4. Attach post-core semantic landmarks (`Extension 4B`)
    attached_landmarks = attach_semantic_landmarks(
        kernel=kernel,
        state=merged_state,
        requests=landmark_requests,
        default_license_filter=license_filter,
        allowed_part_ids=eff_allowed_ids,
        rem_inv=rem_inv,
        foundation_policy=foundation_policy,
    )
    landmark_iids: List[int] = []
    for lm_iid, lm_pid, lm_pose, lm_grounded in attached_landmarks:
        landmark_iids.append(lm_iid)
        trace_steps.append(
            AtomicBuildStep(
                op_type=AtomicOpType.PLACE_PART,
                instance_ids=(lm_iid,),
                part_id=lm_pid,
                pose=lm_pose,
                is_grounded=lm_grounded,
            )
        )

    # 5. Global color allocation if `colored_inventory` is active across the whole scene
    if colored_inventory is not None and merged_state.num_parts > 0:
        # Pool vertex color samples across submodels in world coordinates
        all_v_pts: List[np.ndarray] = []
        all_v_rgb: List[np.ndarray] = []
        for n in ordered_nodes:
            if n.vertex_colors_rgb is not None:
                vp = n.verts_ldu if n.verts_ldu is not None else n.adaptive_target.mesh_surf_pts
                if vp.shape[0] > 0 and n.vertex_colors_rgb.shape[0] == vp.shape[0]:
                    off_arr = np.array([[n.world_offset_ldu[0], n.world_offset_ldu[1], n.world_offset_ldu[2]]], dtype=np.float64)
                    all_v_pts.append(np.asarray(vp, dtype=np.float64) + off_arr)
                    all_v_rgb.append(np.asarray(n.vertex_colors_rgb, dtype=np.float64))
        v_pts_cat = np.vstack(all_v_pts) if all_v_pts else None
        v_rgb_cat = np.vstack(all_v_rgb) if all_v_rgb else None
        alloc_c = allocate_inventory_colors_min_cost(
            merged_state, colored_inventory, verts_ldu=v_pts_cat, vertex_colors_rgb=v_rgb_cat
        )
        # Note: keep subassembly trace intact by only splitting non-subassemblybuilds or skipping splits here
        _ = (alloc_c, apply_chromatic_same_footprint_splits)

    exec_trace = ExecutionTrace(
        steps=tuple(trace_steps),
        allow_subassemblies=any(s.op_type == AtomicOpType.PLACE_SUBASSEMBLY for s in trace_steps),
        allow_temp_supports=False,
        allow_reorientation=False,
        trace_source="compositional_subassembly_realizer",
    )
    setattr(merged_state, "_execution_trace", exec_trace)

    mpd_str = export_compositional_mpd(
        submodel_records=sub_records,
        state=merged_state,
        extra_root_instance_ids=landmark_iids,
        main_model_name="main.ldr",
    )
    if not combined_boxes:
        combined_boxes.append(LDUBox(0, 0, 0, 20, 20, 24))
    comb_tf = SparseTiledTargetField(boxes=tuple(combined_boxes))
    return CompositionalMPDResult(
        assembly=merged_state,
        execution_trace=exec_trace,
        combined_target_field=comb_tf,
        submodel_records=tuple(sub_records),
        mpd_text=mpd_str,
        landmarks_attached=len(landmark_iids),
    )
