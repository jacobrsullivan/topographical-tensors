"""
Route-policy resolution, provenance tracking, and multi-route dispatch for `TopoBrickKernel.realize`
(`Phase 15 Priorities 1, 2 & 3`).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.independent_verifier import compute_target_sha256
from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import (
    EXTENDED_10PART_RECT_KIT,
    STABLETEXT2BRICK_8PART_KIT,
    get_brick_primitive,
    is_known_part_id,
)

NAMED_RECTANGULAR_KITS: Dict[str, Tuple[str, ...]] = {
    "STABLETEXT2BRICK_8PART_KIT": tuple(STABLETEXT2BRICK_8PART_KIT),
    "EXTENDED_10PART_RECT_KIT": tuple(EXTENDED_10PART_RECT_KIT),
}
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import (
    ActiveConnectionFrontier,
    AssemblyState,
    PlacedPart,
    PortPolarity,
    compute_cheap_pre_expansion_port_scores,
)
from topo_brick.kernel.course_solvers import (
    GreedyLayerMergeControl,
    LegolizationBaseline,
    LuoInspiredSeamRefinement,
    _build_blind_grounded_base_pool,
)
from topo_brick.kernel.insertion_corridors import check_monotone_insertion_corridor
from topo_brick.kernel.kernel_contracts import (
    CertStatus,
    CertificateBundle,
    ExecutionTrace,
    GeometryCert,
    InsertionCert,
    KernelRealizeResult,
    KernelResultStatus,
    MinimumBuildRecord,
    PrefixMechanicsCert,
    RealizeRouteProvenance,
    StaticMechanicsCert,
    _get_peak_rss_mb,
    parse_external_record_specs,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField, TargetShapeBoxes


@dataclass(frozen=True, slots=True)
class RealizeRequestSpec:
    """Immutable request specification accepted by `TopoBrickKernel.realize(request_spec=...)` (`Phase 15`)."""

    target_field: SparseTiledTargetField | TargetShapeBoxes
    solver_id: str = "S1_CoverAndRepair"
    allowed_part_ids: Optional[Sequence[str]] = None
    kit_name: str = "STABLETEXT2BRICK_8PART_KIT"
    inventory_counts: Optional[Dict[str, int]] = None
    support_policy: str = "none"
    port_filter_keep_ratio: Optional[float] = None
    m2_filter: Optional[Any] = None
    object_id: str = "blind_target"
    source_dataset: str = "blind_target_field"
    source_row_index: int = -1
    category_id: str = ""
    caption: str = ""
    split: str = "test"
    reference_part_count: int = 0
    foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE
    verify_perturbation: bool = False
    t_target_init_ms: float = 0.0
    forbidden_support_boxes: Tuple[LDUBox, ...] = ()
    surface_refinement: str = "none"
    adaptive_mesh_target: Optional[Any] = None
    macro_block_mode: str = "off"
    part_retrieval: str = "none"
    retrieval_top_k: int = 32
    license_filter: str = "all"
    semantic_style_hint: Optional[str] = None
    vertex_colors_rgb: Optional[np.ndarray] = None
    verts_ldu: Optional[np.ndarray] = None
    colored_inventory: Optional[Any] = None
    hollow_shell_policy: Optional[Any] = None
    subassembly_graph: Optional[Sequence[Any]] = None
    landmark_requests: Sequence[Any] = ()
    view_priority: Optional[Any] = None
    max_wall_clock_s: Optional[float] = None
    deadline_monotonic_s: Optional[float] = None
    ground_plane_z_ldu: float = 0.0
    vertex_region_labels: Optional[np.ndarray] = None
    open_color_palette_mode: str = "solid_54_default"
    symmetry_plane_axis: Optional[str] = None
    symmetry_plane_coord_ldu: Optional[float] = None


ROUTE_CONSUMED_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "compositional_mpd": (
        "subassembly_graph",
        "allowed_part_ids",
        "inventory_counts",
        "colored_inventory",
        "part_retrieval",
        "retrieval_top_k",
        "license_filter",
        "semantic_style_hint",
        "vertex_colors_rgb",
        "verts_ldu",
        "landmark_requests",
        "view_priority",
        "forbidden_support_boxes",
        "support_policy",
        "foundation_policy",
        "ground_plane_z_ldu",
        "deadline_monotonic_s",
        "open_color_palette_mode",
        "vertex_region_labels",
        "symmetry_plane_axis",
        "symmetry_plane_coord_ldu",
        "verify_perturbation",
    ),
    "open_catalog_sculpted": (
        "target_field",
        "adaptive_mesh_target",
        "allowed_part_ids",
        "inventory_counts",
        "colored_inventory",
        "part_retrieval",
        "retrieval_top_k",
        "license_filter",
        "semantic_style_hint",
        "vertex_colors_rgb",
        "verts_ldu",
        "hollow_shell_policy",
        "view_priority",
        "landmark_requests",
        "forbidden_support_boxes",
        "support_policy",
        "foundation_policy",
        "ground_plane_z_ldu",
        "deadline_monotonic_s",
        "open_color_palette_mode",
        "vertex_region_labels",
        "symmetry_plane_axis",
        "symmetry_plane_coord_ldu",
        "verify_perturbation",
    ),
    "adaptive_plate_slope": (
        "target_field",
        "adaptive_mesh_target",
        "allowed_part_ids",
        "inventory_counts",
        "forbidden_support_boxes",
        "support_policy",
        "foundation_policy",
        "ground_plane_z_ldu",
        "deadline_monotonic_s",
        "verify_perturbation",
    ),
    "baseline_layer_merge": (
        "target_field",
        "solver_id",
        "allowed_part_ids",
        "inventory_counts",
        "foundation_policy",
        "ground_plane_z_ldu",
        "deadline_monotonic_s",
        "verify_perturbation",
    ),
    "luo_seam_refinement": (
        "target_field",
        "solver_id",
        "allowed_part_ids",
        "inventory_counts",
        "foundation_policy",
        "ground_plane_z_ldu",
        "deadline_monotonic_s",
        "verify_perturbation",
    ),
    "hierarchical_macro_block": (
        "target_field",
        "macro_block_mode",
        "allowed_part_ids",
        "inventory_counts",
        "forbidden_support_boxes",
        "support_policy",
        "foundation_policy",
        "ground_plane_z_ldu",
        "deadline_monotonic_s",
        "verify_perturbation",
    ),
    "course_by_course_search": (
        "target_field",
        "solver_id",
        "allowed_part_ids",
        "inventory_counts",
        "port_filter_keep_ratio",
        "m2_filter",
        "forbidden_support_boxes",
        "support_policy",
        "foundation_policy",
        "ground_plane_z_ldu",
        "deadline_monotonic_s",
        "verify_perturbation",
    ),
}


def compute_route_option_accounting(
    req: RealizeRequestSpec,
    executed_route_id: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Compute `(consumed_options, ignored_nondefault_options)` for `executed_route_id` (`Phase 16.1`).

    Guarantees that `consumed_options` and `ignored_nondefault_options` are strictly disjoint and
    that every explicitly non-default behavioural option on `req` — including options missing a
    prerequisite on the executed route — appears in `consumed_options` XOR `ignored_nondefault_options`.
    """
    fp_enum = (
        req.foundation_policy
        if isinstance(req.foundation_policy, FoundationPolicy)
        else FoundationPolicy(str(req.foundation_policy))
    )
    has_hollow = req.hollow_shell_policy is not None and bool(
        getattr(req.hollow_shell_policy, "enabled", False)
    )
    is_sculpt_or_comp = executed_route_id in ("open_catalog_sculpted", "compositional_mpd")

    has_matching_vcols_sculpt = False
    if req.vertex_colors_rgb is not None and len(req.vertex_colors_rgb) > 0:
        if req.verts_ldu is not None:
            has_matching_vcols_sculpt = len(req.verts_ldu) == len(req.vertex_colors_rgb)
        elif (
            executed_route_id == "open_catalog_sculpted"
            and req.adaptive_mesh_target is not None
            and getattr(req.adaptive_mesh_target, "mesh_surf_pts", None) is not None
        ):
            has_matching_vcols_sculpt = (
                len(req.adaptive_mesh_target.mesh_surf_pts) == len(req.vertex_colors_rgb)
            )

    has_node_vcols_comp = bool(
        executed_route_id == "compositional_mpd"
        and req.subassembly_graph is not None
        and any(getattr(n, "vertex_colors_rgb", None) is not None for n in req.subassembly_graph)
    )

    has_matching_vreg_sculpt = False
    if req.vertex_region_labels is not None and len(req.vertex_region_labels) > 0:
        if req.verts_ldu is not None:
            has_matching_vreg_sculpt = len(req.verts_ldu) == len(req.vertex_region_labels)
        elif (
            executed_route_id == "open_catalog_sculpted"
            and req.adaptive_mesh_target is not None
            and getattr(req.adaptive_mesh_target, "mesh_surf_pts", None) is not None
        ):
            has_matching_vreg_sculpt = (
                len(req.adaptive_mesh_target.mesh_surf_pts) == len(req.vertex_region_labels)
            )

    has_rect_chromatic = bool(
        (not is_sculpt_or_comp)
        and req.colored_inventory is not None
        and req.verts_ldu is not None
        and req.vertex_colors_rgb is not None
        and len(req.verts_ldu) > 0
        and len(req.verts_ldu) == len(req.vertex_colors_rgb)
    )

    # Tuple of `(option_name, is_nondefault, is_consumed_on_executed_route)`
    option_specs: Tuple[Tuple[str, bool, bool], ...] = (
        ("target_field", False, executed_route_id != "compositional_mpd"),
        ("subassembly_graph", req.subassembly_graph is not None, executed_route_id == "compositional_mpd"),
        (
            "surface_refinement",
            req.surface_refinement != "none",
            (executed_route_id == "open_catalog_sculpted" and req.surface_refinement == "open_catalog_sculpted")
            or (
                executed_route_id == "adaptive_plate_slope"
                and req.surface_refinement == "adaptive_plate_slope"
                and req.adaptive_mesh_target is not None
            ),
        ),
        (
            "adaptive_mesh_target",
            req.adaptive_mesh_target is not None,
            executed_route_id == "adaptive_plate_slope"
            or (executed_route_id == "open_catalog_sculpted" and req.adaptive_mesh_target is not None),
        ),
        ("macro_block_mode", req.macro_block_mode != "off", executed_route_id == "hierarchical_macro_block"),
        (
            "solver_id",
            req.solver_id != "S1_CoverAndRepair",
            executed_route_id in ("course_by_course_search", "baseline_layer_merge", "luo_seam_refinement"),
        ),
        ("allowed_part_ids", req.allowed_part_ids is not None, True),
        (
            "kit_name",
            req.kit_name != "STABLETEXT2BRICK_8PART_KIT",
            (not is_sculpt_or_comp)
            and req.allowed_part_ids is None
            and req.inventory_counts is None
            and req.colored_inventory is None
            and req.kit_name in NAMED_RECTANGULAR_KITS,
        ),
        ("inventory_counts", req.inventory_counts is not None, True),
        (
            "colored_inventory",
            req.colored_inventory is not None,
            is_sculpt_or_comp or req.colored_inventory is not None,
        ),
        (
            "support_policy",
            req.support_policy != "none",
            executed_route_id
            in (
                "compositional_mpd",
                "open_catalog_sculpted",
                "adaptive_plate_slope",
                "hierarchical_macro_block",
                "course_by_course_search",
            ),
        ),
        (
            "port_filter_keep_ratio",
            req.port_filter_keep_ratio is not None,
            executed_route_id == "course_by_course_search",
        ),
        ("m2_filter", req.m2_filter is not None, executed_route_id == "course_by_course_search"),
        ("foundation_policy", fp_enum != FoundationPolicy.CLUTCHED_BASEPLATE, True),
        ("verify_perturbation", bool(req.verify_perturbation), True),
        (
            "forbidden_support_boxes",
            bool(req.forbidden_support_boxes),
            executed_route_id
            in (
                "compositional_mpd",
                "open_catalog_sculpted",
                "adaptive_plate_slope",
                "hierarchical_macro_block",
                "course_by_course_search",
            ),
        ),
        ("part_retrieval", req.part_retrieval != "none", is_sculpt_or_comp),
        (
            "retrieval_top_k",
            int(req.retrieval_top_k) != 32,
            is_sculpt_or_comp and req.part_retrieval == "open_catalog_top_k",
        ),
        ("license_filter", req.license_filter != "all", is_sculpt_or_comp),
        ("semantic_style_hint", req.semantic_style_hint is not None, is_sculpt_or_comp),
        ("hollow_shell_policy", has_hollow, executed_route_id == "open_catalog_sculpted"),
        ("view_priority", req.view_priority is not None, is_sculpt_or_comp),
        ("landmark_requests", bool(req.landmark_requests), is_sculpt_or_comp),
        (
            "vertex_colors_rgb",
            req.vertex_colors_rgb is not None,
            (is_sculpt_or_comp and has_matching_vcols_sculpt) or has_rect_chromatic,
        ),
        (
            "verts_ldu",
            req.verts_ldu is not None,
            (
                is_sculpt_or_comp
                and req.verts_ldu is not None
                and (has_matching_vcols_sculpt or has_matching_vreg_sculpt)
            )
            or has_rect_chromatic,
        ),
        ("vertex_region_labels", req.vertex_region_labels is not None, is_sculpt_or_comp and has_matching_vreg_sculpt),
        (
            "open_color_palette_mode",
            req.open_color_palette_mode != "solid_54_default",
            is_sculpt_or_comp
            and req.colored_inventory is None
            and (has_matching_vcols_sculpt or has_node_vcols_comp),
        ),
        ("symmetry_plane_axis", req.symmetry_plane_axis is not None, is_sculpt_or_comp),
        (
            "symmetry_plane_coord_ldu",
            req.symmetry_plane_coord_ldu is not None,
            is_sculpt_or_comp and req.symmetry_plane_axis is not None,
        ),
        ("max_wall_clock_s", req.max_wall_clock_s is not None, req.max_wall_clock_s is not None),
        ("deadline_monotonic_s", req.deadline_monotonic_s is not None, True),
        ("ground_plane_z_ldu", abs(float(req.ground_plane_z_ldu)) > 1e-9, True),
    )
    consumed = tuple(opt for opt, _is_nd, is_cons in option_specs if is_cons)
    ignored = tuple(opt for opt, is_nd, is_cons in option_specs if is_nd and not is_cons)
    return consumed, ignored


@dataclass(frozen=True, slots=True)
class ResolvedRealizePolicy:
    """Resolved cross-route execution policy computed once at `TopoBrickKernel.realize()` entry (`Phase 15–16`)."""

    candidate_route_id: str
    eff_deadline_monotonic_s: Optional[float]
    world_ground_plane_z_ldu: float
    foundation_policy: FoundationPolicy
    support_policy: str
    license_filter: str
    effective_kit: Tuple[str, ...]
    effective_inventory_counts: Optional[Dict[str, int]]
    min_recall_gate: float
    phase11_sculpt_trigger: bool
    consumed_options: Tuple[str, ...] = ()
    ignored_nondefault_options: Tuple[str, ...] = ()


def resolve_realize_policy(kernel: Any, req: RealizeRequestSpec) -> ResolvedRealizePolicy:
    """Resolve route precedence, shared monotonic deadline, ground datum, kit, and recall threshold."""
    eff_deadline = req.deadline_monotonic_s
    if req.max_wall_clock_s is not None:
        cand_dl = time.monotonic() + max(0.0, float(req.max_wall_clock_s))
        eff_deadline = cand_dl if eff_deadline is None else min(eff_deadline, cand_dl)

    fp_enum = (
        req.foundation_policy
        if isinstance(req.foundation_policy, FoundationPolicy)
        else FoundationPolicy(str(req.foundation_policy))
    )

    eff_init_inv: Optional[Dict[str, int]] = None
    if req.colored_inventory is not None and hasattr(req.colored_inventory, "marginal_mold_counts"):
        eff_init_inv = req.colored_inventory.marginal_mold_counts()
        if req.inventory_counts is not None:
            for k_i, v_i in req.inventory_counts.items():
                eff_init_inv[k_i] = min(eff_init_inv.get(k_i, int(v_i)), int(v_i))
    elif req.inventory_counts is not None:
        eff_init_inv = dict(req.inventory_counts)

    if req.allowed_part_ids is not None:
        kit = tuple(req.allowed_part_ids)
    elif eff_init_inv is not None:
        kit = (
            tuple(sorted(pid for pid, q in eff_init_inv.items() if int(q) > 0 and is_known_part_id(pid)))
            or kernel.default_kit
        )
    elif req.kit_name != "STABLETEXT2BRICK_8PART_KIT" and req.kit_name in NAMED_RECTANGULAR_KITS:
        kit = NAMED_RECTANGULAR_KITS[req.kit_name]
    else:
        kit = kernel.default_kit

    phase11_sculpt_trigger = bool(
        (req.hollow_shell_policy is not None and getattr(req.hollow_shell_policy, "enabled", False))
        or req.view_priority is not None
        or bool(req.landmark_requests)
        or (req.colored_inventory is not None and req.adaptive_mesh_target is not None)
    )

    if req.subassembly_graph is not None:
        route_id = "compositional_mpd"
        min_recall = 0.35
    elif (
        req.surface_refinement == "open_catalog_sculpted"
        or req.part_retrieval == "open_catalog_top_k"
        or (req.surface_refinement == "none" and phase11_sculpt_trigger)
    ):
        route_id = "open_catalog_sculpted"
        min_recall = 0.35
    elif req.surface_refinement == "adaptive_plate_slope" and req.adaptive_mesh_target is not None:
        route_id = "adaptive_plate_slope"
        min_recall = 0.72
    elif req.solver_id in ("Legolization_Baseline", "GreedyLayerMerge_Control"):
        route_id = "baseline_layer_merge"
        min_recall = 0.90
    elif req.solver_id in ("LuoInspiredSeamRefinement", "Luo2015_IterativeSeamRefinement"):
        route_id = "luo_seam_refinement"
        min_recall = 0.90
    else:
        from topo_brick.kernel.adaptive_surface_and_macro_planner import HierarchicalMacroBlockPlanner

        if HierarchicalMacroBlockPlanner.should_activate(req.target_field, req.macro_block_mode):
            route_id = "hierarchical_macro_block"
        else:
            route_id = "course_by_course_search"
        min_recall = 0.90

    consumed, ignored = compute_route_option_accounting(req, route_id)

    return ResolvedRealizePolicy(
        candidate_route_id=route_id,
        eff_deadline_monotonic_s=eff_deadline,
        world_ground_plane_z_ldu=float(req.ground_plane_z_ldu),
        foundation_policy=fp_enum,
        support_policy=str(req.support_policy),
        license_filter=str(req.license_filter),
        effective_kit=kit,
        effective_inventory_counts=eff_init_inv,
        min_recall_gate=min_recall,
        phase11_sculpt_trigger=phase11_sculpt_trigger,
        consumed_options=consumed,
        ignored_nondefault_options=ignored,
    )


def _compute_exact_field_iou(state: AssemblyState, tf: Any) -> float:
    if state.num_parts == 0 or tf is None:
        return 0.0
    t_vol = int(getattr(tf, "total_volume_ldu3", 0))
    if t_vol <= 0:
        return 0.0
    all_b = [b for p in state.placed_parts.values() for b in p.world_boxes]
    if not all_b:
        return 0.0
    asm_vol = int(sum(b.volume_ldu3 for b in all_b))
    inter = int(tf.intersection_volume(all_b))
    union = max(1, asm_vol + t_vol - inter)
    return float(inter / union)


_TARGET_SHA_RING: Dict[int, Tuple[Any, str]] = {}


def _cached_target_sha256(tf: Any) -> str:
    if tf is None:
        return "no_target"
    tid = id(tf)
    hit = _TARGET_SHA_RING.get(tid)
    if hit is not None and hit[0] is tf:
        return hit[1]
    sha = compute_target_sha256(tf)
    if len(_TARGET_SHA_RING) >= 64:
        _TARGET_SHA_RING.clear()
    _TARGET_SHA_RING[tid] = (tf, sha)
    return sha


def _build_route_provenance(
    *,
    executed_route_id: str,
    requested_solver_id: str,
    caller_target_field: Any,
    evaluated_target_field: Any,
    evaluated_target_kind: str,
    min_recall_gate: float,
    state: AssemblyState,
    certs: CertificateBundle,
    adaptive_mesh_target: Optional[Any],
    foundation_policy: FoundationPolicy | str,
    ground_plane_z_ldu: float,
    license_filter: str,
    compositional_mpd_text: Optional[str],
    candidate_route_id: str = "",
    fallback_from_route_id: Optional[str] = None,
    workbench_origin_ldu: Tuple[int, int, int] = (0, 0, 0),
    consumed_options: Tuple[str, ...] = (),
    ignored_nondefault_options: Tuple[str, ...] = (),
) -> RealizeRouteProvenance:
    caller_sha = _cached_target_sha256(caller_target_field)
    eval_sha = (
        caller_sha
        if evaluated_target_field is caller_target_field
        else _cached_target_sha256(evaluated_target_field)
    )
    eval_vol = int(getattr(evaluated_target_field, "total_volume_ldu3", 0))
    if eval_vol <= 0 or state.num_parts == 0:
        eval_iou = 0.0
        eval_rec = 0.0
    else:
        eval_iou = float(certs.geometry.target_iou)
        eval_rec = float(certs.geometry.target_recall)
    caller_iou = (
        eval_iou
        if eval_sha == caller_sha
        else _compute_exact_field_iou(state, caller_target_field)
    )
    mesh_iou: Optional[float] = None
    if adaptive_mesh_target is not None and hasattr(adaptive_mesh_target, "occ_5x5x2"):
        if state.num_parts == 0:
            mesh_iou = 0.0
        else:
            from topo_brick.kernel.adaptive_surface_and_macro_planner import evaluate_exact_5x5x2_mesh_iou

            mesh_iou = float(evaluate_exact_5x5x2_mesh_iou(adaptive_mesh_target, state))

    fp_str = (
        foundation_policy.value
        if isinstance(foundation_policy, FoundationPolicy)
        else str(foundation_policy)
    )
    return RealizeRouteProvenance(
        effective_route_id=str(executed_route_id),
        requested_solver_id=str(requested_solver_id),
        caller_target_sha256=caller_sha,
        evaluated_target_sha256=eval_sha,
        target_was_refined_or_transformed=(eval_sha != caller_sha),
        evaluated_target_kind=str(evaluated_target_kind),
        min_recall_threshold=float(min_recall_gate),
        achieved_evaluated_recall=eval_rec,
        achieved_evaluated_iou=eval_iou,
        achieved_caller_target_iou=caller_iou,
        achieved_mesh_5x5x2_iou=mesh_iou,
        foundation_policy=fp_str,
        ground_plane_z_ldu=float(ground_plane_z_ldu),
        license_filter=str(license_filter),
        recommended_mpd_source=(
            "compositional_mpd_text" if compositional_mpd_text is not None else "assembly.to_ldraw_mpd"
        ),
        candidate_route_id=str(candidate_route_id or executed_route_id),
        fallback_from_route_id=fallback_from_route_id,
        workbench_origin_ldu=tuple(int(c) for c in workbench_origin_ldu),
        consumed_options=tuple(consumed_options),
        ignored_nondefault_options=tuple(ignored_nondefault_options),
    )


def execute_resolved_realize(
    self: Any,
    req: RealizeRequestSpec,
    policy: Optional[ResolvedRealizePolicy] = None,
) -> KernelRealizeResult:
    """Execute `TopoBrickKernel.realize` under resolved cross-route policy (`Phase 15`)."""
    _ = policy
    target_field = req.target_field
    solver_id = req.solver_id
    allowed_part_ids = req.allowed_part_ids
    kit_name = req.kit_name
    support_policy = req.support_policy
    port_filter_keep_ratio = req.port_filter_keep_ratio
    m2_filter = req.m2_filter
    object_id = req.object_id
    source_dataset = req.source_dataset
    source_row_index = req.source_row_index
    category_id = req.category_id
    caption = req.caption
    split = req.split
    reference_part_count = req.reference_part_count
    foundation_policy = req.foundation_policy
    verify_perturbation = req.verify_perturbation
    t_target_init_ms = req.t_target_init_ms
    forbidden_support_boxes = req.forbidden_support_boxes
    adaptive_mesh_target = req.adaptive_mesh_target
    macro_block_mode = req.macro_block_mode
    part_retrieval = req.part_retrieval
    retrieval_top_k = req.retrieval_top_k
    license_filter = req.license_filter
    semantic_style_hint = req.semantic_style_hint
    vertex_colors_rgb = req.vertex_colors_rgb
    verts_ldu = req.verts_ldu
    colored_inventory = req.colored_inventory
    hollow_shell_policy = req.hollow_shell_policy
    subassembly_graph = req.subassembly_graph
    landmark_requests = req.landmark_requests
    view_priority = req.view_priority
    ground_plane_z_ldu = req.ground_plane_z_ldu
    vertex_region_labels = req.vertex_region_labels
    open_color_palette_mode = req.open_color_palette_mode
    symmetry_plane_axis = req.symmetry_plane_axis
    symmetry_plane_coord_ldu = req.symmetry_plane_coord_ldu
    t0 = time.perf_counter()
    cpu_t0 = time.process_time()
    pol = policy if policy is not None else resolve_realize_policy(self, req)
    eff_deadline: Optional[float] = pol.eff_deadline_monotonic_s

    def _budget_hit() -> bool:
        return eff_deadline is not None and time.monotonic() >= eff_deadline

    lp_calls_0 = int(getattr(self.validator, "lp_solver_calls", 0))
    wit_hits_0 = int(getattr(self.validator, "witness_reuse_hits", 0))
    max_fr_ports = 0
    tot_cands_gen = 0
    tot_conf_edges = 0
    eff_init_inv = pol.effective_inventory_counts
    kit = pol.effective_kit
    for pid in kit:
        if not is_known_part_id(pid):
            empty_st = AssemblyState()
            certs_u = self.certify(empty_st, verify_prefix=False, verify_insertion=False)
            rec_u = MinimumBuildRecord(
                object_id=object_id,
                source_dataset=source_dataset,
                source_row_index=source_row_index,
                category_id=category_id,
                caption=caption,
                split=split,
                solver_id=solver_id,
                kit_name=kit_name,
                status=KernelResultStatus.UNSUPPORTED_SPEC.value,
                num_parts_placed=0,
                reference_part_count=reference_part_count,
                num_courses=0,
                target_iou=0.0,
                target_precision=0.0,
                target_recall=0.0,
                collision_free=True,
                protrusion_collision_free=True,
                static_stable=True,
                prefix_stable=True,
                monotone_insertable=True,
                peak_utilization=0.0,
                wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
                peak_rss_mb=_get_peak_rss_mb(),
                no_good_cuts_used=0,
                certificate_summary=certs_u.summary_dict(),
            )
            return KernelRealizeResult(
                status=KernelResultStatus.UNSUPPORTED_SPEC,
                assembly=empty_st,
                certificates=certs_u,
                build_record=rec_u,
                realization_mode="blind",
            )

    target_vol = int(target_field.total_volume_ldu3)
    bounds = target_field.occupied_bounds_ldu()
    xmin, ymin, zmin, xmax, ymax, zmax = bounds
    if (target_vol <= 0 or zmax <= zmin) and subassembly_graph is None and adaptive_mesh_target is None:
        empty_st = AssemblyState()
        certs_e = self.certify(empty_st, target_field=target_field, verify_prefix=False, verify_insertion=False)
        rec_e = MinimumBuildRecord(
            object_id=object_id,
            source_dataset=source_dataset,
            source_row_index=source_row_index,
            category_id=category_id,
            caption=caption,
            split=split,
            solver_id=solver_id,
            kit_name=kit_name,
            status=KernelResultStatus.PROVEN_INFEASIBLE.value,
            num_parts_placed=0,
            reference_part_count=reference_part_count,
            num_courses=0,
            target_iou=0.0,
            target_precision=0.0,
            target_recall=0.0,
            collision_free=True,
            protrusion_collision_free=True,
            static_stable=True,
            prefix_stable=True,
            monotone_insertable=True,
            peak_utilization=0.0,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            peak_rss_mb=_get_peak_rss_mb(),
            no_good_cuts_used=0,
            certificate_summary=certs_e.summary_dict(),
        )
        return KernelRealizeResult(
            status=KernelResultStatus.PROVEN_INFEASIBLE,
            assembly=empty_st,
            certificates=certs_e,
            build_record=rec_e,
            realization_mode="blind",
        )

    # Determine vertical course bands blindly from `target_field`
    has_plate_parts = any(
        (get_brick_primitive(pid).hull_boxes[0].zmax - get_brick_primitive(pid).hull_boxes[0].zmin) == 8
        for pid in kit
        if get_brick_primitive(pid).hull_boxes
    )
    z_bands: List[Tuple[int, int]] = []
    z_cursor = zmin
    while z_cursor < zmax:
        if has_plate_parts:
            v0 = target_field.band_volume_ldu3(z_cursor, z_cursor + 8)
            v1 = target_field.band_volume_ldu3(z_cursor + 8, z_cursor + 16)
            v2 = target_field.band_volume_ldu3(z_cursor + 16, z_cursor + 24)
            if not (v0 == v1 == v2):
                if v0 > 0:
                    z_bands.append((z_cursor, 8))
                z_cursor += 8
                continue
        if target_field.band_volume_ldu3(z_cursor, z_cursor + 24) > 0:
            z_bands.append((z_cursor, 24))
        z_cursor += 24

    nogood_total = 0
    ext_support_bricks = 0
    ext_support_vol = 0
    t_target_acc = float(t_target_init_ms) + (time.perf_counter() - t0) * 1000.0
    t_cands_acc = 0.0
    t_sel_acc = 0.0
    t_rep_acc = 0.0
    rem_inv: Optional[Dict[str, int]] = dict(eff_init_inv) if eff_init_inv is not None else None

    def _active_kit() -> Tuple[str, ...]:
        if rem_inv is None:
            return kit
        return tuple(pid for pid in kit if rem_inv.get(pid, 0) > 0)

    precomputed_certs: Optional[CertificateBundle] = None
    t_cert_pre_ms = 0.0
    eff_cert_target_field: SparseTiledTargetField | TargetShapeBoxes = target_field
    min_recall_gate: float = float(pol.min_recall_gate)
    open_cat_diag: Optional[Any] = None
    comp_mpd_text: Optional[str] = None
    explicit_exec_trace: Optional[ExecutionTrace] = None
    executed_route_id: str = pol.candidate_route_id
    fallback_from_route_id: Optional[str] = None
    workbench_origin_ldu: Tuple[int, int, int] = (0, 0, 0)
    evaluated_target_kind: str = "caller_target_field"
    eff_ad_for_prov: Optional[Any] = adaptive_mesh_target
    comp_had_nonempty_input: bool = True

    from topo_brick.kernel.adaptive_surface_and_macro_planner import (
        AdaptiveSurfaceTarget,
        HierarchicalMacroBlockPlanner,
        realize_adaptive_plate_slope_surface,
    )
    from topo_brick.kernel.workbench_frame import UnsupportedWorkbenchFrameError

    def _return_unsupported_frame(reason: str) -> KernelRealizeResult:
        empty_st = AssemblyState()
        certs_u = self.certify(
            empty_st,
            target_field=target_field,
            verify_prefix=False,
            verify_insertion=False,
            foundation_policy=pol.foundation_policy,
            ground_plane_z_ldu=ground_plane_z_ldu,
        )
        rec_u = MinimumBuildRecord(
            object_id=object_id,
            source_dataset=source_dataset,
            source_row_index=source_row_index,
            category_id=category_id,
            caption=caption,
            split=split,
            solver_id=solver_id,
            kit_name=kit_name,
            status=KernelResultStatus.UNSUPPORTED_SPEC.value,
            num_parts_placed=0,
            reference_part_count=reference_part_count,
            num_courses=0,
            target_iou=0.0,
            target_precision=0.0,
            target_recall=0.0,
            collision_free=True,
            protrusion_collision_free=True,
            static_stable=True,
            prefix_stable=True,
            monotone_insertable=True,
            peak_utilization=0.0,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            peak_rss_mb=_get_peak_rss_mb(),
            no_good_cuts_used=0,
            certificate_summary=certs_u.summary_dict(),
        )
        rp_u = _build_route_provenance(
            executed_route_id=pol.candidate_route_id,
            requested_solver_id=solver_id,
            caller_target_field=target_field,
            evaluated_target_field=target_field,
            evaluated_target_kind="caller_target_field",
            min_recall_gate=pol.min_recall_gate,
            state=empty_st,
            certs=certs_u,
            adaptive_mesh_target=adaptive_mesh_target,
            foundation_policy=pol.foundation_policy,
            ground_plane_z_ldu=ground_plane_z_ldu,
            license_filter=license_filter,
            compositional_mpd_text=None,
            candidate_route_id=pol.candidate_route_id,
            consumed_options=pol.consumed_options,
            ignored_nondefault_options=pol.ignored_nondefault_options,
        )
        return KernelRealizeResult(
            status=KernelResultStatus.UNSUPPORTED_SPEC,
            assembly=empty_st,
            certificates=certs_u,
            build_record=rec_u,
            realization_mode="blind",
            stopping_reason=f"unrepresentable_workbench_frame:{reason}",
            route_provenance=rp_u,
        )

    # Single-owner dispatch strictly via `pol.candidate_route_id` (`Phase 16 Decision 2`)
    if _budget_hit():
        state = AssemblyState()
        extra_waves = 0
    elif pol.candidate_route_id == "compositional_mpd":
        from topo_brick.kernel.subassembly_realizer import realize_compositional_mpd

        t_s0 = time.perf_counter()
        executed_route_id = "compositional_mpd"
        evaluated_target_kind = "compositional_union_5x5x6"
        if subassembly_graph is None:
            return _return_unsupported_frame("missing_subassembly_graph")
        eff_comp_top_k = int(retrieval_top_k) if part_retrieval == "open_catalog_top_k" else 32
        comp_res = realize_compositional_mpd(
            kernel=self,
            subassembly_graph=subassembly_graph,
            allowed_part_ids=allowed_part_ids,
            part_retrieval=(
                "open_catalog_top_k"
                if (part_retrieval == "open_catalog_top_k" or allowed_part_ids is None)
                else part_retrieval
            ),
            retrieval_top_k=eff_comp_top_k,
            license_filter=license_filter,
            semantic_style_hint=semantic_style_hint,
            inventory_counts=rem_inv,
            colored_inventory=colored_inventory,
            landmark_requests=landmark_requests,
            view_priority=view_priority,
            support_policy=support_policy if support_policy != "none" else "minimal_permanent_pillars",
            foundation_policy=pol.foundation_policy,
            deadline_monotonic_s=eff_deadline,
            ground_plane_z_ldu=ground_plane_z_ldu,
            world_forbidden_support_boxes=forbidden_support_boxes,
            open_color_palette_mode=open_color_palette_mode,
            verts_ldu=verts_ldu,
            vertex_colors_rgb=vertex_colors_rgb,
            vertex_region_labels=vertex_region_labels,
            symmetry_plane_axis=symmetry_plane_axis,
            symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
        )
        state = comp_res.assembly
        explicit_exec_trace = comp_res.execution_trace
        comp_mpd_text = comp_res.mpd_text
        open_cat_diag = comp_res.open_catalog_diagnostics
        comp_had_nonempty_input = bool(comp_res.had_nonempty_input_submodels)
        eff_cert_target_field = comp_res.combined_target_field
        t_sel_acc += (time.perf_counter() - t_s0) * 1000.0
        extra_waves = 0
    elif pol.candidate_route_id == "open_catalog_sculpted":
        from topo_brick.kernel.open_catalog_realizer import realize_open_catalog_sculpted

        t_s0 = time.perf_counter()
        eff_ad = adaptive_mesh_target
        try:
            if eff_ad is None:
                eff_ad = AdaptiveSurfaceTarget.from_target_boxes(
                    getattr(target_field, "boxes", ()),
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    hollow_shell_policy=hollow_shell_policy,
                    forbidden_support_boxes=forbidden_support_boxes,
                )
                # Caller-target retention sanity guard when hollowing & openings are absent (`Decision 1`)
                if (
                    not (hollow_shell_policy is not None and getattr(hollow_shell_policy, "enabled", False))
                    and not forbidden_support_boxes
                    and int(target_field.total_volume_ldu3) > 0
                ):
                    synthesized_tf = eff_ad.to_fine_target_field()
                    retained_vol = int(target_field.intersection_volume(synthesized_tf.boxes))
                    if retained_vol < int(0.999 * float(target_field.total_volume_ldu3)):
                        raise UnsupportedWorkbenchFrameError(
                            f"caller_target_volume_loss:{retained_vol}<{int(target_field.total_volume_ldu3)}"
                        )
            workbench_origin_ldu = tuple(int(c) for c in getattr(eff_ad, "origin_ldu", (0, 0, 0)))
            eff_ad_for_prov = eff_ad
            eff_oc_top_k = int(retrieval_top_k) if part_retrieval == "open_catalog_top_k" else 32
            state, fine_oc_tf, open_cat_diag = realize_open_catalog_sculpted(
                kernel=self,
                coarse_target_field=target_field,
                adaptive_target=eff_ad,
                allowed_part_ids=allowed_part_ids,
                part_retrieval=(
                    "open_catalog_top_k"
                    if (part_retrieval == "open_catalog_top_k" or allowed_part_ids is None)
                    else part_retrieval
                ),
                retrieval_top_k=eff_oc_top_k,
                license_filter=license_filter,
                semantic_style_hint=semantic_style_hint,
                vertex_colors_rgb=vertex_colors_rgb,
                verts_ldu=verts_ldu,
                inventory_counts=rem_inv,
                support_policy=(
                    support_policy
                    if support_policy != "none" or not pol.phase11_sculpt_trigger
                    else "minimal_permanent_pillars"
                ),
                foundation_policy=pol.foundation_policy,
                forbidden_support_boxes=forbidden_support_boxes,
                colored_inventory=colored_inventory,
                hollow_shell_policy=hollow_shell_policy,
                view_priority=view_priority,
                landmark_requests=landmark_requests,
                deadline_monotonic_s=eff_deadline,
                ground_plane_z_ldu=ground_plane_z_ldu,
                vertex_region_labels=vertex_region_labels,
                open_color_palette_mode=open_color_palette_mode,
                symmetry_plane_axis=symmetry_plane_axis,
                symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
            )
        except UnsupportedWorkbenchFrameError as exc_wb:
            return _return_unsupported_frame(str(exc_wb))
        executed_route_id = "open_catalog_sculpted"
        evaluated_target_kind = (
            "hollow_crust_5x5x6"
            if (hollow_shell_policy is not None and getattr(hollow_shell_policy, "enabled", False))
            else "adaptive_fine_5x5x6"
        )
        eff_cert_target_field = fine_oc_tf
        t_cands_acc += float(open_cat_diag.retrieval_result.query_wall_ms)
        t_sel_acc += max(
            0.0,
            (time.perf_counter() - t_s0) * 1000.0 - float(open_cat_diag.retrieval_result.query_wall_ms),
        )
        extra_waves = 0
    elif pol.candidate_route_id == "adaptive_plate_slope":
        t_s0 = time.perf_counter()
        executed_route_id = "adaptive_plate_slope"
        evaluated_target_kind = "adaptive_fine_5x5x6"
        if adaptive_mesh_target is None:
            return _return_unsupported_frame("missing_adaptive_mesh_target")
        try:
            workbench_origin_ldu = tuple(
                int(c) for c in getattr(adaptive_mesh_target, "origin_ldu", (0, 0, 0))
            )
            state, _ = realize_adaptive_plate_slope_surface(
                kernel=self,
                coarse_target_field=target_field,
                adaptive_target=adaptive_mesh_target,
                allowed_part_ids=kit,
                inventory_counts=rem_inv,
                support_policy=support_policy,
                foundation_policy=pol.foundation_policy,
                forbidden_support_boxes=forbidden_support_boxes,
                deadline_monotonic_s=eff_deadline,
                ground_plane_z_ldu=ground_plane_z_ldu,
            )
        except UnsupportedWorkbenchFrameError as exc_wb:
            return _return_unsupported_frame(str(exc_wb))
        fine_tf = adaptive_mesh_target.to_fine_target_field()
        if fine_tf.total_volume_ldu3 > 0:
            eff_cert_target_field = fine_tf
        t_sel_acc += (time.perf_counter() - t_s0) * 1000.0
        extra_waves = 0
    elif pol.candidate_route_id == "baseline_layer_merge":
        t_s0 = time.perf_counter()
        executed_route_id = "baseline_layer_merge"
        state, _ = GreedyLayerMergeControl.realize_blind(
            target_field=target_field,
            allowed_part_ids=kit,
            validator=self.validator,
            deadline_monotonic_s=eff_deadline,
            ground_plane_z_ldu=ground_plane_z_ldu,
        )
        t_sel_acc += (time.perf_counter() - t_s0) * 1000.0
        extra_waves = 0
    elif pol.candidate_route_id == "luo_seam_refinement":
        t_s0 = time.perf_counter()
        executed_route_id = "luo_seam_refinement"
        state, _ = LuoInspiredSeamRefinement.realize_blind(
            target_field=target_field,
            allowed_part_ids=kit,
            validator=self.validator,
            deadline_monotonic_s=eff_deadline,
            ground_plane_z_ldu=ground_plane_z_ldu,
        )
        t_sel_acc += (time.perf_counter() - t_s0) * 1000.0
        extra_waves = 0
    else:
        macro_succeeded = False
        if pol.candidate_route_id == "hierarchical_macro_block":
            t_m0 = time.perf_counter()
            m_state, m_cands, m_blocks = HierarchicalMacroBlockPlanner.realize_macro_blocks(
                kernel=self,
                target_field=target_field,
                allowed_part_ids=kit,
                inventory_counts=rem_inv,
                support_policy=support_policy,
                foundation_policy=pol.foundation_policy,
                forbidden_support_boxes=forbidden_support_boxes,
                deadline_monotonic_s=eff_deadline,
                ground_plane_z_ldu=ground_plane_z_ldu,
            )
            t_m_pack = (time.perf_counter() - t_m0) * 1000.0
            if m_state is not None and m_state.num_parts > 0 and not _budget_hit():
                t_mc0 = time.perf_counter()
                m_certs = self.certify(
                    m_state,
                    target_field=target_field,
                    verify_prefix=True,
                    verify_insertion=True,
                    foundation_policy=foundation_policy,
                    verify_perturbation=verify_perturbation,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=eff_deadline,
                )
                t_mc_dt = (time.perf_counter() - t_mc0) * 1000.0
                if m_certs.overall_certified and m_certs.geometry.target_recall >= pol.min_recall_gate:
                    state = m_state
                    precomputed_certs = m_certs
                    t_cert_pre_ms = t_mc_dt
                    tot_cands_gen += m_cands
                    extra_waves = m_blocks
                    t_sel_acc += t_m_pack
                    macro_succeeded = True
                    executed_route_id = "hierarchical_macro_block"
                elif macro_block_mode == "force":
                    state = m_state
                    precomputed_certs = m_certs
                    t_cert_pre_ms = t_mc_dt
                    tot_cands_gen += m_cands
                    extra_waves = m_blocks
                    t_sel_acc += t_m_pack
                    macro_succeeded = True
                    executed_route_id = "hierarchical_macro_block"

        if not macro_succeeded:
            if pol.candidate_route_id == "hierarchical_macro_block":
                fallback_from_route_id = "hierarchical_macro_block"
            executed_route_id = "course_by_course_search"
            state = AssemblyState()
        if (not macro_succeeded) and z_bands and not _budget_hit() and macro_block_mode != "force":
            # 1. Blind grounded base course tiling at z_bands[0]
            z0_base, dz_base = z_bands[0]
            t_tg = time.perf_counter()
            base_target = target_field.slice_z_band(z0_base, z0_base + dz_base)
            t_target_acc += (time.perf_counter() - t_tg) * 1000.0
            t_cd = time.perf_counter()
            base_pool = _build_blind_grounded_base_pool(
                base_target, z0=z0_base, dz=dz_base, allowed_part_ids=_active_kit()
            )
            t_cands_acc += (time.perf_counter() - t_cd) * 1000.0
            max_fr_ports = max(max_fr_ports, len(base_pool.port_to_candidates))
            tot_cands_gen += len(base_pool)
            tot_conf_edges += sum(len(s) for s in base_pool.conflict_adjacency) // 2
            t_sl = time.perf_counter()
            sel_base, _ = self._select_from_pool(
                solver_id,
                state,
                base_pool,
                base_target,
                inventory_counts=rem_inv,
                foundation_policy=foundation_policy,
                ground_plane_z_ldu=ground_plane_z_ldu,
                deadline_monotonic_s=eff_deadline,
            )
            t_sel_acc += (time.perf_counter() - t_sl) * 1000.0
            for cid in sel_base:
                cand_b = base_pool.candidates[cid]
                try:
                    state.place_root_part(cand_b.part_id, cand_b.pose, is_grounded=True)
                    if rem_inv is not None:
                        rem_inv[cand_b.part_id] = max(0, rem_inv.get(cand_b.part_id, 0) - 1)
                except ValueError:
                    continue

            # 2. Blind upward course-by-course sweep across z_bands[1:]
            for z0, dz in z_bands[1:]:
                if _budget_hit():
                    break
                cur_kit = _active_kit()
                if not cur_kit:
                    break
                t_tg = time.perf_counter()
                course_target = target_field.slice_z_band(z0, z0 + dz)
                t_target_acc += (time.perf_counter() - t_tg) * 1000.0
                t_cd = time.perf_counter()
                frontier = state.get_active_frontier(min_z_ldu=z0)
                max_fr_ports = max(max_fr_ports, len(frontier))
                if len(frontier) == 0:
                    t_cands_acc += (time.perf_counter() - t_cd) * 1000.0
                    continue
                if port_filter_keep_ratio is not None and len(frontier) > 6:
                    max_ports = max(4, int(round(len(frontier) * port_filter_keep_ratio)))
                    if m2_filter is not None:
                        p_scores = m2_filter.score_ports(state, frontier, course_target)
                    else:
                        p_scores = compute_cheap_pre_expansion_port_scores(state, frontier, course_target)
                    pool = state.generate_candidate_pool_selective(
                        port_priority_scores=p_scores,
                        max_expanded_ports=max_ports,
                        allowed_part_ids=cur_kit,
                        frontier=frontier,
                    )
                else:
                    # Filter frontier strictly to ports whose outward 4-LDU probe lies inside course_target
                    in_target_ports = tuple(
                        ep
                        for ep in frontier.ports
                        if course_target.contains_point(
                            ep.world_frame.transform_point(
                                (0, 0, 4 if ep.polarity == PortPolarity.STUD else -4)
                            )
                        )
                    )
                    if not in_target_ports:
                        t_cands_acc += (time.perf_counter() - t_cd) * 1000.0
                        continue
                    sub_fr = ActiveConnectionFrontier(
                        ports=in_target_ports,
                        port_index_by_id=frontier.port_index_by_id,
                    )
                    pool = state.generate_candidate_pool(
                        allowed_part_ids=cur_kit,
                        frontier=sub_fr,
                        target_shape=course_target,
                        min_target_precision=0.95,
                    )
                t_cands_acc += (time.perf_counter() - t_cd) * 1000.0
                tot_cands_gen += len(pool)
                tot_conf_edges += sum(len(s) for s in pool.conflict_adjacency) // 2

                t_sl = time.perf_counter()
                sel, ng = self._select_from_pool(
                    solver_id,
                    state,
                    pool,
                    course_target,
                    inventory_counts=rem_inv,
                    full_target=target_field,
                    z0_course=z0,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=eff_deadline,
                )
                t_sel_acc += (time.perf_counter() - t_sl) * 1000.0
                nogood_total += ng
                if sel:
                    state.commit_course(pool, sel)
                    if rem_inv is not None:
                        for cid in sel:
                            pid_u = pool.candidates[cid].part_id
                            rem_inv[pid_u] = max(0, rem_inv.get(pid_u, 0) - 1)
                    # Immediately attach any newly bridged top-hung apron / lateral extension in
                    # `[max(zmin, z0 - 24), min(zmax, z0 + dz + 8))` BEFORE higher courses build a roof over it!
                    z_sub_hi = min(zmax, z0 + dz + 8)
                    placed_vol_below = sum(
                        b.volume_ldu3
                        for p in state.placed_parts.values()
                        for b in p.world_boxes
                        if b.zmax <= z_sub_hi
                    )
                    cum_band_vol = (
                        target_field.band_volume_ldu3(zmin, z_sub_hi)
                        if hasattr(target_field, "band_volume_ldu3")
                        else target_field.slice_z_band(zmin, z_sub_hi).total_volume_ldu3
                    )
                    local_band = (
                        target_field.slice_z_band(max(zmin, z0 - 24), z_sub_hi)
                        if placed_vol_below < int(0.999 * cum_band_vol)
                        else None
                    )
                    for _sub in range(2 if local_band is not None else 0):
                        cur_kit_s = _active_kit()
                        if not cur_kit_s:
                            break
                        fr_s = state.get_active_frontier(min_z_ldu=max(zmin, z0 - 24))
                        cand_ports_s = []
                        for ep in fr_s.ports:
                            if ep.polarity == PortPolarity.STUD and ep.world_frame.z_axis == (0, 0, 1) and ep.world_frame.pos[2] >= z0 + dz:
                                continue
                            pt4_s = ep.world_frame.transform_point(
                                (0, 0, 4 if ep.polarity == PortPolarity.STUD else -4)
                            )
                            if local_band.contains_point(pt4_s) and not state._point_inside_any_box(
                                pt4_s, ignore_instance=ep.global_port_id[0]
                            ):
                                cand_ports_s.append(ep)
                        sub_ports_s = tuple(cand_ports_s)
                        if not sub_ports_s:
                            break
                        pool_s = state.generate_candidate_pool(
                            allowed_part_ids=cur_kit_s,
                            frontier=ActiveConnectionFrontier(
                                ports=sub_ports_s, port_index_by_id=fr_s.port_index_by_id
                            ),
                            target_shape=local_band,
                            min_target_precision=0.95,
                        )
                        if len(pool_s) == 0:
                            break
                        sel_s, ng_s = self._select_from_pool(
                            solver_id,
                            state,
                            pool_s,
                            local_band,
                            inventory_counts=rem_inv,
                            foundation_policy=foundation_policy,
                            ground_plane_z_ldu=ground_plane_z_ldu,
                            deadline_monotonic_s=eff_deadline,
                        )
                        nogood_total += ng_s
                        cx_mid_s = 0.5 * (xmin + xmax)
                        cy_mid_s = 0.5 * (ymin + ymax)
                        ordered_sel_s = sorted(
                            sel_s,
                            key=lambda cs: (
                                (pool_s.candidates[cs].pose.pos[0] - cx_mid_s) ** 2
                                + (pool_s.candidates[cs].pose.pos[1] - cy_mid_s) ** 2
                            ),
                        )
                        committed_s = 0
                        for cs in ordered_sel_s:
                            cand_s = pool_s.candidates[cs]
                            if check_monotone_insertion_corridor(
                                state, cand_s.part_id, cand_s.pose
                            ):
                                try:
                                    state.commit_candidate(cand_s)
                                    committed_s += 1
                                    if rem_inv is not None:
                                        rem_inv[cand_s.part_id] = max(
                                            0, rem_inv.get(cand_s.part_id, 0) - 1
                                        )
                                except ValueError:
                                    pass
                        if committed_s == 0:
                            break

            # 3. Blind multi-wave frontier-completion sweep (realizes downward-hanging aprons,
            #    inverted arches, corbelled cantilevers, and SNOT side attachments blindly)
            def _run_frontier_completion_waves(max_waves: int = 16) -> int:
                nonlocal nogood_total
                waves_used = 0
                for _ in range(max_waves):
                    if _budget_hit():
                        break
                    cur_kit_w = _active_kit()
                    if not cur_kit_w:
                        break
                    placed_vol_now = sum(
                        b.volume_ldu3 for p in state.placed_parts.values() for b in p.world_boxes
                    )
                    # Check target intersection if external supports exist, else placed_vol_now
                    if ext_support_bricks > 0:
                        inter_now = target_field.intersection_volume(
                            [b for p in state.placed_parts.values() for b in p.world_boxes]
                        )
                    else:
                        inter_now = placed_vol_now
                    if inter_now >= int(0.999 * target_vol):
                        break
                    full_fr = state.get_active_frontier()
                    cand_ports = []
                    for ep in full_fr.ports:
                        pt4 = ep.world_frame.transform_point(
                            (0, 0, 4 if ep.polarity == PortPolarity.STUD else -4)
                        )
                        if target_field.contains_point(pt4) and not state._point_inside_any_box(
                            pt4, ignore_instance=ep.global_port_id[0]
                        ):
                            cand_ports.append(ep)
                    if not cand_ports:
                        break
                    sub_fr_w = ActiveConnectionFrontier(
                        ports=tuple(cand_ports),
                        port_index_by_id=full_fr.port_index_by_id,
                    )
                    pool_w = state.generate_candidate_pool(
                        allowed_part_ids=cur_kit_w,
                        frontier=sub_fr_w,
                        target_shape=target_field,
                        min_target_precision=0.95,
                    )
                    if len(pool_w) == 0:
                        break
                    sel_w, ng_w = self._select_from_pool(
                        solver_id,
                        state,
                        pool_w,
                        target_field,
                        inventory_counts=rem_inv,
                        foundation_policy=foundation_policy,
                        ground_plane_z_ldu=ground_plane_z_ldu,
                        deadline_monotonic_s=eff_deadline,
                    )
                    nogood_total += ng_w
                    # Commit fill-in placements inside-out sequentially so earlier placements
                    # never seal off a sibling brick's monotone insertion corridor in `certify()`.
                    cx_mid_w = 0.5 * (xmin + xmax)
                    cy_mid_w = 0.5 * (ymin + ymax)
                    ordered_sel_w = sorted(
                        sel_w,
                        key=lambda cw_i: (
                            (pool_w.candidates[cw_i].pose.pos[0] - cx_mid_w) ** 2
                            + (pool_w.candidates[cw_i].pose.pos[1] - cy_mid_w) ** 2
                        ),
                    )
                    committed_w = 0
                    for cid_w in ordered_sel_w:
                        cw = pool_w.candidates[cid_w]
                        c_z0 = min(b.zmin for b in cw.world_boxes)
                        has_bottom_support = ext_support_bricks > 0 and any(
                            max(b.zmax for b in state.placed_parts[gp[0]].world_boxes) <= c_z0
                            for gp in cw.consumed_ports
                            if gp[0] in state.placed_parts
                        )
                        if has_bottom_support or check_monotone_insertion_corridor(
                            state, cw.part_id, cw.pose
                        ):
                            try:
                                state.commit_candidate(cw)
                                committed_w += 1
                                if rem_inv is not None:
                                    rem_inv[cw.part_id] = max(
                                        0, rem_inv.get(cw.part_id, 0) - 1
                                    )
                            except ValueError:
                                pass
                    if committed_w == 0:
                        break
                    waves_used += 1
                return waves_used

            t_rp0 = time.perf_counter()
            extra_waves = _run_frontier_completion_waves()

            # 4. Optional Lexicographic Minimal-Volume Permanent Support Pillars (`Claim B`)
            if support_policy == "minimal_permanent_pillars" and "3005" in kit and not _budget_hit():
                for _pillar_round in range(12):
                    if _budget_hit():
                        break
                    placed_boxes_all = [b for p in state.placed_parts.values() for b in p.world_boxes]
                    inter_cur = target_field.intersection_volume(placed_boxes_all)
                    eq_now = self.validator.evaluate(
                        state,
                        foundation_policy=foundation_policy,
                        ground_plane_z_ldu=ground_plane_z_ldu,
                        deadline_monotonic_s=eff_deadline,
                    )
                    if eq_now.budget_exceeded:
                        break
                    if inter_cur >= int(0.995 * target_vol) and eq_now.is_stable:
                        break
                    # Find either:
                    # (a) uncovered target stud cells (x0, y0, z_u), OR
                    # (b) already-covered cantilevered target cells with an exposed bottom ANTI_STUD
                    #     at (x0 + 10, y0 + 10, z_u) when `not eq_now.is_stable`!
                    grounded_centers = [
                        (0.5 * (b.xmin + b.xmax), 0.5 * (b.ymin + b.ymax))
                        for p in state.placed_parts.values()
                        if p.is_grounded
                        for b in p.world_boxes
                    ]
                    best_col: Optional[Tuple[float, int, int, int]] = None
                    for z_u, dz_u in z_bands:
                        if z_u <= zmin:
                            continue
                        for x0 in range(xmin, xmax, 20):
                            for y0 in range(ymin, ymax, 20):
                                pt_c = (x0 + 10, y0 + 10, z_u + dz_u // 2)
                                if not target_field.contains_point(pt_c):
                                    continue
                                already_covered = state._point_inside_any_box(pt_c, ignore_instance=-1)
                                if already_covered:
                                    if eq_now.is_stable:
                                        continue
                                    co_ports = state._ports_by_pos.get((x0 + 10, y0 + 10, z_u), ())
                                    has_open_antistud = any(
                                        gp not in state._matched_ports and pol == PortPolarity.ANTI_STUD
                                        for gp, _, pol, _ in co_ports
                                    )
                                    if not has_open_antistud:
                                        continue
                                # Respect declared architectural openings / forbidden support zones (`Gate 2`)
                                if forbidden_support_boxes:
                                    col_span = LDUBox(x0, y0, zmin, x0 + 20, y0 + 20, z_u)
                                    if any(col_span.intersects(fb) for fb in forbidden_support_boxes):
                                        continue
                                # Verify vertical column [zmin, z_u) has at least one empty slot
                                # and no solid non-mateable obstacle at z_u - 24
                                has_empty_slot = any(
                                    not state.check_boxes_collision(
                                        (LDUBox(x0, y0, z_p, x0 + 20, y0 + 20, z_p + 24),)
                                    )
                                    for z_p in range(zmin, z_u, 24)
                                )
                                if not has_empty_slot:
                                    continue
                                cx_f, cy_f = float(x0 + 10), float(y0 + 10)
                                dist_g = (
                                    min((cx_f - gx) ** 2 + (cy_f - gy) ** 2 for gx, gy in grounded_centers)
                                    if grounded_centers
                                    else 0.0
                                )
                                uncovered_bonus = 1e6 if not already_covered else 0.0
                                score_col = uncovered_bonus + dist_g - 0.1 * float(z_u - zmin)
                                if best_col is None or score_col > best_col[0]:
                                    best_col = (score_col, x0, y0, z_u)
                    if best_col is None:
                        break
                    _, px0, py0, pz_top = best_col
                    placed_any_pillar = False
                    for z_p in range(zmin, pz_top, 24):
                        p_box = LDUBox(px0, py0, z_p, px0 + 20, py0 + 20, z_p + 24)
                        if state.check_boxes_collision((p_box,)):
                            continue
                        if rem_inv is not None and rem_inv.get("3005", 0) <= 0:
                            break
                        try:
                            state.place_root_part(
                                "3005",
                                LDUFrame(0, (px0 + 10, py0 + 10, z_p)),
                                is_grounded=(z_p <= zmin),
                            )
                            placed_any_pillar = True
                            if not target_field.contains_point((px0 + 10, py0 + 10, z_p + 12)):
                                ext_support_bricks += 1
                                ext_support_vol += 9600
                            if rem_inv is not None:
                                rem_inv["3005"] = max(0, rem_inv.get("3005", 0) - 1)
                        except ValueError:
                            break
                    if not placed_any_pillar:
                        break
                    extra_waves += _run_frontier_completion_waves()
            # Canonicalize execution order bottom-up (`zmin` ascending, grounded/supported first)
            # when support pillars added lower-course bricks out of `z` order
            if ext_support_bricks > 0 and state.num_parts > 1:
                ordered = AssemblyState()
                placed_orig_ids: Set[int] = set()
                orig_adj = state._part_adj
                remaining = list(state.placed_parts.values())
                remaining.sort(
                    key=lambda p: (
                        min(b.zmin for b in p.world_boxes) if p.world_boxes else p.pose.pos[2],
                        0 if p.is_grounded else 1,
                        p.instance_id,
                    )
                )
                made_progress = True
                while remaining and made_progress:
                    made_progress = False
                    # Strictly place ONLY the lowest-zmin eligible wave so top-hung bricks at z_low
                    # are inserted immediately after their z_high bridge brick, before higher courses seal them!
                    active_wave_z: Optional[int] = None
                    next_remaining: List[PlacedPart] = []
                    for p_rem in remaining:
                        p_z0 = min(b.zmin for b in p_rem.world_boxes) if p_rem.world_boxes else p_rem.pose.pos[2]
                        if active_wave_z is not None and p_z0 > active_wave_z:
                            next_remaining.append(p_rem)
                            continue
                        connected_now = bool(
                            p_rem.is_grounded
                            or (orig_adj.get(p_rem.instance_id, set()) & placed_orig_ids)
                        )
                        if not connected_now:
                            next_remaining.append(p_rem)
                            continue
                        if not check_monotone_insertion_corridor(ordered, p_rem.part_id, p_rem.pose):
                            next_remaining.append(p_rem)
                            continue
                        try:
                            ordered.place_root_part(
                                p_rem.part_id,
                                p_rem.pose,
                                is_grounded=p_rem.is_grounded,
                                color_id=p_rem.color_id,
                            )
                            placed_orig_ids.add(p_rem.instance_id)
                            active_wave_z = p_z0
                            made_progress = True
                        except ValueError:
                            next_remaining.append(p_rem)
                    remaining = next_remaining
                if not remaining and ordered.num_parts == state.num_parts:
                    ord_certs = self.certify(
                        ordered,
                        target_field=target_field,
                        verify_prefix=True,
                        verify_insertion=True,
                        foundation_policy=foundation_policy,
                        verify_perturbation=verify_perturbation,
                    )
                    if ord_certs.overall_certified:
                        state = ordered
                        precomputed_certs = ord_certs
            t_rep_acc += (time.perf_counter() - t_rp0) * 1000.0
        elif not macro_succeeded:
            extra_waves = 0

    # Phase 11 Ext 1: If `colored_inventory` was passed to a non-sculpted rectangular solver path
    # (`open_cat_diag is None and subassembly_graph is None`), run two-zone chromatic allocation + splits now!
    if (
        colored_inventory is not None
        and open_cat_diag is None
        and subassembly_graph is None
        and state.num_parts > 0
    ):
        from topo_brick.kernel.colored_inventory_allocator import (
            allocate_inventory_colors_min_cost,
            apply_chromatic_same_footprint_splits,
        )

        fp_ch = (
            foundation_policy
            if isinstance(foundation_policy, FoundationPolicy)
            else FoundationPolicy(str(foundation_policy))
        )
        alloc_rect = allocate_inventory_colors_min_cost(
            state, colored_inventory, verts_ldu=verts_ldu, vertex_colors_rgb=vertex_colors_rgb
        )
        alloc_rect = apply_chromatic_same_footprint_splits(
            state,
            alloc_rect,
            colored_inventory,
            verts_ldu,
            vertex_colors_rgb,
            self.validator,
            fp_ch,
        )
        chrom_summary_obj = alloc_rect
        precomputed_certs = None
    else:
        chrom_summary_obj = getattr(open_cat_diag, "chromatic_allocation", None)

    t_cert0 = time.perf_counter()
    if precomputed_certs is not None and (eff_init_inv is not None or colored_inventory is not None):
        # Re-certify with finite inventory bounds if macro-block shortcut ran without `InventoryCert`
        precomputed_certs = None
    if precomputed_certs is not None:
        certs = precomputed_certs
    elif _budget_hit():
        fp_val_b = (
            foundation_policy.value
            if isinstance(foundation_policy, FoundationPolicy)
            else str(foundation_policy)
        )
        certs = CertificateBundle(
            geometry=GeometryCert(
                status=CertStatus.UNKNOWN,
                collision_free=True,
                protrusion_collision_free=True,
                all_ports_compatible=True,
                state_integrity_verified=False,
                target_iou=0.0,
                target_precision=0.0,
                target_recall=0.0,
            ),
            static_mechanics=StaticMechanicsCert(
                status=CertStatus.UNKNOWN,
                is_stable=False,
                peak_utilization=float("inf"),
                safety_margin=-float("inf"),
                weakest_contact=None,
                moment_reference="budget_exceeded",
                asymmetric_bearing_active=True,
                cold_rebuild_consistent=False,
                solve_time_ms=0.0,
            ),
            prefix_mechanics=PrefixMechanicsCert(CertStatus.UNKNOWN, 0, 0, None, float("inf")),
            insertion=InsertionCert(CertStatus.UNKNOWN, 0, 0, ()),
            foundation_policy=fp_val_b,
            execution_trace=explicit_exec_trace,
            ground_plane_z_ldu=float(ground_plane_z_ldu),
        )
    else:
        certs = self.certify(
            state,
            target_field=eff_cert_target_field,
            verify_prefix=True,
            verify_insertion=True,
            execution_trace=explicit_exec_trace,
            foundation_policy=foundation_policy,
            verify_perturbation=verify_perturbation,
            inventory_counts=eff_init_inv,
            colored_inventory=colored_inventory,
            ground_plane_z_ldu=ground_plane_z_ldu,
            deadline_monotonic_s=eff_deadline,
        )
    if chrom_summary_obj is not None:
        from dataclasses import replace as dc_replace

        certs = dc_replace(
            certs,
            inventory=dc_replace(
                certs.inventory,
                skin_mean_delta_e00=float(chrom_summary_obj.skin_mean_delta_e00),
                fifo_baseline_skin_mean_delta_e00=float(chrom_summary_obj.fifo_baseline_skin_mean_delta_e00),
                core_parts_count=int(chrom_summary_obj.num_core_parts),
                chromatic_splits_accepted=int(chrom_summary_obj.splits_accepted),
            ),
        )
    t_cert_total = t_cert_pre_ms if precomputed_certs is not None else (time.perf_counter() - t_cert0) * 1000.0
    t_mech_ms = float(certs.static_mechanics.solve_time_ms)
    t_exec_ms = max(0.0, t_cert_total - t_mech_ms)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 + float(t_target_init_ms)
    coverage_ok = certs.geometry.target_recall >= min_recall_gate and state.num_parts > 0
    forb_clear = True
    if forbidden_support_boxes and state.num_parts > 0:
        forb_clear = not any(
            wb.intersects(fb)
            for pl_f in state.placed_parts.values()
            for wb in pl_f.world_boxes
            for fb in forbidden_support_boxes
        )
    budget_expired_final = _budget_hit()
    stopping_reason = ""
    if certs.overall_certified and coverage_ok and forb_clear and not budget_expired_final:
        status = KernelResultStatus.COMPLETED_CERTIFIED
    elif budget_expired_final:
        status = KernelResultStatus.BUDGET_EXCEEDED
        stopping_reason = "wall_clock_deadline_exceeded"
    elif int(eff_cert_target_field.total_volume_ldu3) <= 0 and state.num_parts == 0:
        if executed_route_id == "compositional_mpd" and not comp_had_nonempty_input:
            status = KernelResultStatus.UNSUPPORTED_SPEC
            stopping_reason = "empty_compositional_submodel_target"
        elif executed_route_id == "compositional_mpd":
            status = KernelResultStatus.PROVEN_INFEASIBLE
            stopping_reason = "empty_compositional_target_after_forbidden_exclusions"
        else:
            status = KernelResultStatus.PROVEN_INFEASIBLE
            stopping_reason = "empty_evaluated_target_after_exclusions"
    elif not forb_clear:
        status = KernelResultStatus.PARTIAL_OR_TIMEOUT
        stopping_reason = "forbidden_support_box_intrusion"
    else:
        status = KernelResultStatus.PARTIAL_OR_TIMEOUT
    shortfall_rep: Optional[Any] = None
    if (
        eff_init_inv is not None
        and status != KernelResultStatus.COMPLETED_CERTIFIED
        and status != KernelResultStatus.BUDGET_EXCEEDED
    ):
        from topo_brick.kernel.colored_inventory_allocator import build_inventory_shortfall_report

        used_molds_cur: Dict[str, int] = {}
        for p_u in state.placed_parts.values():
            used_molds_cur[p_u.part_id] = used_molds_cur.get(p_u.part_id, 0) + 1
        rem_molds_cur = {
            pid_k: max(0, int(q_k) - used_molds_cur.get(pid_k, 0))
            for pid_k, q_k in eff_init_inv.items()
        }
        shortfall_rep = build_inventory_shortfall_report(
            required_target_volume_ldu3=int(eff_cert_target_field.total_volume_ldu3),
            state=state,
            initial_inventory_counts=eff_init_inv,
            remaining_inventory_counts=rem_molds_cur,
            colored_inventory=colored_inventory,
        )
        if not shortfall_rep.is_volume_sufficient:
            status = KernelResultStatus.PROVEN_INFEASIBLE
    rem_budget_final: Optional[float] = (
        max(0.0, eff_deadline - time.monotonic()) if eff_deadline is not None else None
    )
    total_courses = len(z_bands) + extra_waves
    n_tiles = int(getattr(target_field, "num_allocated_tiles", 0))
    tile_mib = float(n_tiles * (614400 * 4 + 76800)) / (1024.0 * 1024.0)
    build_rec = MinimumBuildRecord(
        object_id=object_id,
        source_dataset=source_dataset,
        source_row_index=source_row_index,
        category_id=category_id,
        caption=caption,
        split=split,
        solver_id=solver_id,
        kit_name=kit_name,
        status=status.value,
        num_parts_placed=state.num_parts,
        reference_part_count=reference_part_count if reference_part_count > 0 else state.num_parts,
        num_courses=total_courses,
        target_iou=certs.geometry.target_iou,
        target_precision=certs.geometry.target_precision,
        target_recall=certs.geometry.target_recall,
        collision_free=certs.geometry.collision_free,
        protrusion_collision_free=certs.geometry.protrusion_collision_free,
        static_stable=certs.static_mechanics.is_stable,
        prefix_stable=(certs.prefix_mechanics.status == CertStatus.PASS),
        monotone_insertable=(certs.insertion.status == CertStatus.PASS),
        peak_utilization=certs.static_mechanics.peak_utilization,
        wall_clock_ms=elapsed_ms,
        peak_rss_mb=_get_peak_rss_mb(),
        no_good_cuts_used=nogood_total,
        certificate_summary=certs.summary_dict(),
        t_target_ms=t_target_acc,
        t_candidates_ms=t_cands_acc,
        t_selection_ms=t_sel_acc,
        t_mechanics_ms=t_mech_ms,
        t_execution_checks_ms=t_exec_ms,
        t_repair_ms=t_rep_acc,
        active_target_tiles=n_tiles,
        target_tile_memory_mib=tile_mib,
        cpu_time_ms=(time.process_time() - cpu_t0) * 1000.0,
        max_frontier_ports=max_fr_ports,
        total_candidates_generated=tot_cands_gen,
        total_conflict_edges=tot_conf_edges,
        lp_solver_calls=max(0, int(getattr(self.validator, "lp_solver_calls", 0)) - lp_calls_0),
        witness_reuse_hits=max(0, int(getattr(self.validator, "witness_reuse_hits", 0)) - wit_hits_0),
    )
    eff_consumed, eff_ignored = compute_route_option_accounting(req, executed_route_id)
    route_prov = _build_route_provenance(
        executed_route_id=executed_route_id,
        requested_solver_id=solver_id,
        caller_target_field=target_field,
        evaluated_target_field=eff_cert_target_field,
        evaluated_target_kind=evaluated_target_kind,
        min_recall_gate=min_recall_gate,
        state=state,
        certs=certs,
        adaptive_mesh_target=eff_ad_for_prov,
        foundation_policy=foundation_policy,
        ground_plane_z_ldu=ground_plane_z_ldu,
        license_filter=license_filter,
        compositional_mpd_text=comp_mpd_text,
        candidate_route_id=pol.candidate_route_id,
        fallback_from_route_id=fallback_from_route_id,
        workbench_origin_ldu=workbench_origin_ldu,
        consumed_options=eff_consumed,
        ignored_nondefault_options=eff_ignored,
    )
    return KernelRealizeResult(
        status=status,
        assembly=state,
        certificates=certs,
        build_record=build_rec,
        no_good_cuts_added=nogood_total,
        courses_executed=total_courses,
        external_support_bricks_added=ext_support_bricks,
        external_support_volume_ldu3=ext_support_vol,
        realization_mode="blind",
        open_catalog_diagnostics=open_cat_diag,
        evaluated_target_field=eff_cert_target_field,
        inventory_shortfall=shortfall_rep,
        compositional_mpd_text=comp_mpd_text,
        stopping_reason=stopping_reason,
        remaining_budget_s=rem_budget_final,
        route_provenance=route_prov,
    )


def execute_realize_external_record(
    self,
    record: Dict[str, Any],
    solver_id: str = "S1_CoverAndRepair",
    allowed_part_ids: Sequence[str] = STABLETEXT2BRICK_8PART_KIT,
    kit_name: str = "STABLETEXT2BRICK_8PART_KIT",
    port_filter_keep_ratio: Optional[float] = None,
    m2_filter: Optional[Any] = None,
    mode: str = "blind",
    inventory_counts: Optional[Dict[str, int]] = None,
    support_policy: str = "none",
) -> KernelRealizeResult:
    """
    Realize and certify an external LDraw target assembly under the scoped 1-CPU-core
    contract using the specified solver (`S1_CoverAndRepair`, `S2_PairSeededLocalSearch`,
    `S3_RecurrentBatchRefiner`, `S4_D3PMBatchSampler`, or `Legolization_Baseline`).
    By default (`mode="blind"`), parses `record["ldr_text"]` ONLY to construct the voxelized
    `SparseTiledTargetField` and delegates 100% blindly to `self.realize(target_field, ...)`.
    Passing `mode="reference_assisted"` runs the legacy reference-course-assisted path for
    explicit side-by-side comparison.
    """
    t0 = time.perf_counter()
    ldr_text = str(record["ldr_text"])
    allowed_set = set(allowed_part_ids)
    specs, report = parse_external_record_specs(
        ldr_text, allowed_part_ids=allowed_set
    )

    if report.unsupported_valid_part_lines > 0 or len(specs) == 0:
        empty_state = AssemblyState()
        certs = self.certify(empty_state, verify_prefix=False, verify_insertion=False)
        rec_out = MinimumBuildRecord(
            object_id=str(record.get("object_id", "unknown")),
            source_dataset=str(record.get("source_dataset", "AvaLovelace/StableText2Brick")),
            source_row_index=int(record.get("source_row_index", -1)),
            category_id=str(record.get("category_id", "")),
            caption=str(record.get("caption", "")),
            split=str(record.get("split", "test")),
            solver_id=solver_id,
            kit_name=kit_name,
            status=KernelResultStatus.UNSUPPORTED_SPEC.value,
            num_parts_placed=0,
            reference_part_count=len(specs),
            num_courses=0,
            target_iou=0.0,
            target_precision=0.0,
            target_recall=0.0,
            collision_free=True,
            protrusion_collision_free=True,
            static_stable=True,
            prefix_stable=True,
            monotone_insertable=True,
            peak_utilization=0.0,
            wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
            peak_rss_mb=_get_peak_rss_mb(),
            no_good_cuts_used=0,
            certificate_summary=certs.summary_dict(),
        )
        return KernelRealizeResult(
            status=KernelResultStatus.UNSUPPORTED_SPEC,
            assembly=empty_state,
            certificates=certs,
            build_record=rec_out,
            realization_mode=mode,
        )

    all_target_boxes: List[LDUBox] = []
    specs_by_z: Dict[int, List[Any]] = {}
    for sp in specs:
        prim = get_brick_primitive(sp.part_id)
        wbs = tuple(b.transform(sp.pose) for b in prim.hull_boxes)
        all_target_boxes.extend(wbs)
        if mode in ("reference_assisted", "reference_replay"):
            z0 = min(b.zmin for b in wbs)
            specs_by_z.setdefault(z0, []).append(sp)

    t_tf0 = time.perf_counter()
    target_field = SparseTiledTargetField(boxes=tuple(all_target_boxes))
    t_tf_ms = (time.perf_counter() - t_tf0) * 1000.0

    if mode == "reference_replay":
        # S0_Reference_Replay: Pure LDraw reference replay without solver optimization (`R8-WS3`)
        z_courses_r = sorted(specs_by_z.keys())
        min_z_r = z_courses_r[0]
        st_rep = AssemblyState()
        for z0_r in z_courses_r:
            for sp_r in specs_by_z[z0_r]:
                try:
                    st_rep.place_root_part(
                        sp_r.part_id, sp_r.pose, is_grounded=(z0_r <= min_z_r)
                    )
                except ValueError:
                    continue
        t_c0 = time.perf_counter()
        certs_r = self.certify(
            st_rep, target_field=target_field, verify_prefix=True, verify_insertion=True
        )
        t_c_ms = (time.perf_counter() - t_c0) * 1000.0
        elapsed_r = (time.perf_counter() - t0) * 1000.0
        cov_r = certs_r.geometry.target_recall >= 0.90 and st_rep.num_parts > 0
        stat_r = (
            KernelResultStatus.COMPLETED_CERTIFIED
            if (certs_r.overall_certified and cov_r)
            else KernelResultStatus.PARTIAL_OR_TIMEOUT
        )
        n_tiles_r = int(target_field.num_allocated_tiles)
        br_r = MinimumBuildRecord(
            object_id=str(record.get("object_id", "unknown")),
            source_dataset=str(record.get("source_dataset", "AvaLovelace/StableText2Brick")),
            source_row_index=int(record.get("source_row_index", -1)),
            category_id=str(record.get("category_id", "")),
            caption=str(record.get("caption", "")),
            split=str(record.get("split", "test")),
            solver_id="S0_Reference_Replay",
            kit_name=kit_name,
            status=stat_r.value,
            num_parts_placed=st_rep.num_parts,
            reference_part_count=len(specs),
            num_courses=len(z_courses_r),
            target_iou=certs_r.geometry.target_iou,
            target_precision=certs_r.geometry.target_precision,
            target_recall=certs_r.geometry.target_recall,
            collision_free=certs_r.geometry.collision_free,
            protrusion_collision_free=certs_r.geometry.protrusion_collision_free,
            static_stable=certs_r.static_mechanics.is_stable,
            prefix_stable=(certs_r.prefix_mechanics.status == CertStatus.PASS),
            monotone_insertable=(certs_r.insertion.status == CertStatus.PASS),
            peak_utilization=certs_r.static_mechanics.peak_utilization,
            wall_clock_ms=elapsed_r,
            peak_rss_mb=_get_peak_rss_mb(),
            no_good_cuts_used=0,
            certificate_summary=certs_r.summary_dict(),
            t_target_ms=t_tf_ms,
            t_candidates_ms=0.0,
            t_selection_ms=0.0,
            t_mechanics_ms=float(certs_r.static_mechanics.solve_time_ms),
            t_execution_checks_ms=max(0.0, t_c_ms - float(certs_r.static_mechanics.solve_time_ms)),
            t_repair_ms=0.0,
            active_target_tiles=n_tiles_r,
            target_tile_memory_mib=float(n_tiles_r * (614400 * 4 + 76800)) / (1024.0 * 1024.0),
        )
        return KernelRealizeResult(
            status=stat_r,
            assembly=st_rep,
            certificates=certs_r,
            build_record=br_r,
            courses_executed=len(z_courses_r),
            realization_mode="reference_replay",
        )

    if mode != "reference_assisted":
        return self.realize(
            target_field=target_field,
            solver_id=solver_id,
            allowed_part_ids=allowed_part_ids,
            kit_name=kit_name,
            inventory_counts=inventory_counts,
            support_policy=support_policy,
            port_filter_keep_ratio=port_filter_keep_ratio,
            m2_filter=m2_filter,
            object_id=str(record.get("object_id", "unknown")),
            source_dataset=str(record.get("source_dataset", "AvaLovelace/StableText2Brick")),
            source_row_index=int(record.get("source_row_index", -1)),
            category_id=str(record.get("category_id", "")),
            caption=str(record.get("caption", "")),
            split=str(record.get("split", "test")),
            reference_part_count=len(specs),
            t_target_init_ms=t_tf_ms,
        )

    z_courses = sorted(specs_by_z.keys())
    min_z = z_courses[0]
    nogood_total = 0

    if solver_id == "Legolization_Baseline":
        state, _ = LegolizationBaseline.realize_blind(
            target_field=target_field, allowed_part_ids=allowed_part_ids, validator=self.validator
        )
    else:
        state = AssemblyState()
        for sp in specs_by_z[min_z]:
            try:
                state.place_root_part(sp.part_id, sp.pose, is_grounded=True)
            except ValueError:
                pass

        deferred_specs: List[Any] = []
        for z0 in z_courses[1:]:
            course_specs = specs_by_z[z0]
            course_boxes = tuple(
                b.transform(sp.pose)
                for sp in course_specs
                for b in get_brick_primitive(sp.part_id).hull_boxes
            )
            course_target = TargetShapeBoxes(boxes=course_boxes)
            course_parts = tuple(sorted({sp.part_id for sp in course_specs}))

            frontier = state.get_active_frontier(min_z_ldu=z0 - 24)
            if port_filter_keep_ratio is not None and len(frontier) > 6:
                max_ports = max(4, int(round(len(frontier) * port_filter_keep_ratio)))
                if m2_filter is not None:
                    p_scores = m2_filter.score_ports(state, frontier, course_target)
                else:
                    p_scores = compute_cheap_pre_expansion_port_scores(state, frontier, course_target)
                pool = state.generate_candidate_pool_selective(
                    port_priority_scores=p_scores,
                    max_expanded_ports=max_ports,
                    allowed_part_ids=course_parts,
                    frontier=frontier,
                )
            else:
                pool = state.generate_candidate_pool(
                    allowed_part_ids=course_parts, frontier=frontier
                )

            sel, ng = self._select_from_pool(
                solver_id, state, pool, course_target, max_bricks=len(course_specs) + 2
            )
            nogood_total += ng
            if sel:
                state.commit_course(pool, sel)

            pending_to_place = list(course_specs) + deferred_specs
            deferred_specs = []
            made_prog = True
            while made_prog and pending_to_place:
                made_prog = False
                next_pending: List[Any] = []
                depths_cur = state.compute_support_depths()
                for sp in pending_to_place:
                    prim = get_brick_primitive(sp.part_id)
                    wbs = tuple(b.transform(sp.pose) for b in prim.hull_boxes)
                    if state.check_boxes_collision(wbs):
                        continue
                    mpairs, has_clash = state._find_matching_contacts_for_pose(prim, sp.pose)
                    connected_now = any(depths_cur.get(gpid[0], 999) < 999 for _, gpid in mpairs)
                    if not has_clash and connected_now:
                        try:
                            iid = state.place_root_part(sp.part_id, sp.pose, is_grounded=False)
                            depths_cur[iid] = 1
                            made_prog = True
                        except ValueError:
                            pass
                    else:
                        next_pending.append(sp)
                pending_to_place = next_pending
            deferred_specs = pending_to_place

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    certs = self.certify(state, target_field=target_field, verify_prefix=True, verify_insertion=True)
    coverage_ok = certs.geometry.target_recall >= 0.90 and state.num_parts > 0
    status = (
        KernelResultStatus.COMPLETED_CERTIFIED
        if (certs.overall_certified and coverage_ok)
        else KernelResultStatus.PARTIAL_OR_TIMEOUT
    )
    build_rec = MinimumBuildRecord(
        object_id=str(record.get("object_id", "unknown")),
        source_dataset=str(record.get("source_dataset", "AvaLovelace/StableText2Brick")),
        source_row_index=int(record.get("source_row_index", -1)),
        category_id=str(record.get("category_id", "")),
        caption=str(record.get("caption", "")),
        split=str(record.get("split", "test")),
        solver_id=solver_id,
        kit_name=kit_name,
        status=status.value,
        num_parts_placed=state.num_parts,
        reference_part_count=len(specs),
        num_courses=len(z_courses),
        target_iou=certs.geometry.target_iou,
        target_precision=certs.geometry.target_precision,
        target_recall=certs.geometry.target_recall,
        collision_free=certs.geometry.collision_free,
        protrusion_collision_free=certs.geometry.protrusion_collision_free,
        static_stable=certs.static_mechanics.is_stable,
        prefix_stable=(certs.prefix_mechanics.status == CertStatus.PASS),
        monotone_insertable=(certs.insertion.status == CertStatus.PASS),
        peak_utilization=certs.static_mechanics.peak_utilization,
        wall_clock_ms=elapsed_ms,
        peak_rss_mb=_get_peak_rss_mb(),
        no_good_cuts_used=nogood_total,
        certificate_summary=certs.summary_dict(),
    )
    return KernelRealizeResult(
        status=status,
        assembly=state,
        certificates=certs,
        build_record=build_rec,
        no_good_cuts_added=nogood_total,
        courses_executed=len(z_courses),
        realization_mode="reference_assisted",
    )
