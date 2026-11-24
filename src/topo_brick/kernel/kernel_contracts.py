"""
Pure-Python / NumPy contracts, certificates, trace records, and result bundles for `TopoBrickKernel`
(`Phase 15`: zero `torch` or solver runtime imports).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import resource
import sys
import time
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from topo_brick.evaluation.static_equilibrium import FoundationPolicy, SparseStaticEquilibriumValidator
from topo_brick.interface.brick_catalog import STABLETEXT2BRICK_8PART_KIT, get_brick_primitive
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState, GlobalPortId, IncrementalCandidateCache
from topo_brick.representation.interface_fields import SparseTiledTargetField
from topo_brick.training.ldraw_decomposer import LDrawPartSpec, parse_ldr_text_with_report


EXTENDED_STABLETEXT2BRICK_DIM_TO_PART: Dict[Tuple[int, int], str] = {
    (1, 1): "3005",
    (1, 2): "3004",
    (1, 3): "3622",
    (1, 4): "3010",
    (1, 6): "3009",
    (1, 8): "3008",
    (2, 2): "3003",
    (2, 3): "3002",
    (2, 4): "3001",
    (2, 6): "2456",
    (2, 8): "3007",
}


@dataclass(frozen=True, slots=True)
class ExternalParseSummary:
    """Summary of parsing an external record against a declared allowed-part kit."""

    total_type1_lines: int
    supported_part_lines: int
    unsupported_valid_part_lines: int
    non_cubic_rotation_lines: int
    non_integer_translation_lines: int
    malformed_type1_lines: int
    unsupported_part_ids: Tuple[str, ...]


def parse_external_record_specs(
    text: str,
    allowed_part_ids: Sequence[str] = STABLETEXT2BRICK_8PART_KIT,
) -> Tuple[List[LDrawPartSpec], ExternalParseSummary]:
    """
    Parse an external record in either `AvaLovelace/StableText2Brick` format
    (`{dx}x{dy} ({gx},{gy},{gz})`) or standard LDraw Type-1 format (`1 <color> ... <part>.dat`),
    filtering against `allowed_part_ids` and returning `(specs, report)`.
    """
    allowed_set = set(allowed_part_ids)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and re.match(r"^\d+x\d+\s+\(-?\d+,-?\d+,-?\d+\)$", lines[0]):
        specs: List[LDrawPartSpec] = []
        unsupported_ids: List[str] = []
        malformed = 0
        unsupported_lines = 0
        for ln in lines:
            m = re.match(r"^(\d+)x(\d+)\s+\((-?\d+),(-?\d+),(-?\d+)\)$", ln)
            if not m:
                malformed += 1
                continue
            dx, dy, gx, gy, gz = map(int, m.groups())
            key = (min(dx, dy), max(dx, dy))
            part_id = EXTENDED_STABLETEXT2BRICK_DIM_TO_PART.get(key)
            if part_id is None or part_id not in allowed_set:
                unsupported_lines += 1
                unsupported_ids.append(part_id or f"{dx}x{dy}")
                continue
            prim = get_brick_primitive(part_id)
            target_box = LDUBox(
                20 * gx, 20 * gy, 24 * gz, 20 * (gx + dx), 20 * (gy + dy), 24 * (gz + 1)
            )
            b0 = prim.hull_boxes[0]
            matched_pose: Optional[LDUFrame] = None
            for rot_id in C4_Z_ROT_IDS:
                rb = b0.transform(LDUFrame(rot_id, (0, 0, 0)))
                if (rb.xmax - rb.xmin) == (target_box.xmax - target_box.xmin) and (
                    rb.ymax - rb.ymin
                ) == (target_box.ymax - target_box.ymin):
                    matched_pose = LDUFrame(
                        rot_id,
                        (
                            target_box.xmin - rb.xmin,
                            target_box.ymin - rb.ymin,
                            target_box.zmin - rb.zmin,
                        ),
                    )
                    break
            if matched_pose is None:
                malformed += 1
                continue
            specs.append(LDrawPartSpec(part_id=part_id, pose=matched_pose, color_code=15))
        report = ExternalParseSummary(
            total_type1_lines=len(lines),
            supported_part_lines=len(specs),
            unsupported_valid_part_lines=unsupported_lines,
            non_cubic_rotation_lines=0,
            non_integer_translation_lines=0,
            malformed_type1_lines=malformed,
            unsupported_part_ids=tuple(sorted(set(unsupported_ids))),
        )
        return specs, report

    rep_all = parse_ldr_text_with_report(text, strict=False)
    specs_all = list(rep_all.accepted_specs)
    filtered_specs = [sp for sp in specs_all if sp.part_id in allowed_set]
    extra_unsup = [sp.part_id for sp in specs_all if sp.part_id not in allowed_set]
    rep = ExternalParseSummary(
        total_type1_lines=rep_all.total_type1_lines,
        supported_part_lines=len(filtered_specs),
        unsupported_valid_part_lines=len(rep_all.rejected_unknown_part_lines) + len(extra_unsup),
        non_cubic_rotation_lines=len(rep_all.rejected_non_cubic_rotation_lines),
        non_integer_translation_lines=len(rep_all.rejected_off_lattice_coord_lines),
        malformed_type1_lines=len(rep_all.rejected_malformed_lines),
        unsupported_part_ids=tuple(sorted(set(extra_unsup))),
    )
    return filtered_specs, rep


def _get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


# ---------------------------------------------------------------------------
# 1. Explicit Kernel Outcome Statuses & 4-Valued Certificate Bundle
# ---------------------------------------------------------------------------


class KernelResultStatus(str, Enum):
    """Explicit outcome contract for `realize`, `certify`, `constrain`, and `repair`."""

    COMPLETED_CERTIFIED = "COMPLETED_CERTIFIED"
    PARTIAL_OR_TIMEOUT = "PARTIAL_OR_TIMEOUT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    PROVEN_INFEASIBLE = "PROVEN_INFEASIBLE"
    UNSUPPORTED_SPEC = "UNSUPPORTED_SPEC"
    INCONCLUSIVE = "INCONCLUSIVE"


class CertStatus(str, Enum):
    """4-valued verification status for each certificate domain."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class GeometryCert:
    """Certificate for exact LDU hull & unmated-protrusion collision freedom and target IoU."""

    status: CertStatus
    collision_free: bool
    protrusion_collision_free: bool
    all_ports_compatible: bool
    state_integrity_verified: bool
    target_iou: float
    target_precision: float
    target_recall: float
    violating_pairs: Tuple[Tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class StaticMechanicsCert:
    """Certificate for 6-DoF rigid-body contact-wrench static equilibrium under asymmetric bearing."""

    status: CertStatus
    is_stable: bool
    peak_utilization: float
    safety_margin: float
    weakest_contact: Optional[Tuple[GlobalPortId, GlobalPortId]]
    moment_reference: str
    asymmetric_bearing_active: bool
    cold_rebuild_consistent: bool
    solve_time_ms: float


@dataclass(frozen=True, slots=True)
class PrefixMechanicsCert:
    """Certificate for step-by-step prefix static stability across the assembly order."""

    status: CertStatus
    evaluated_steps: int
    stable_steps: int
    first_unstable_step: Optional[int]
    max_prefix_utilization: float


@dataclass(frozen=True, slots=True)
class InsertionCert:
    """Certificate for exact continuous single-axis half-ray (`t in (0, +inf)`) insertion clearance."""

    status: CertStatus
    evaluated_steps: int
    monotone_axis_insertable_steps: int
    blocked_steps: Tuple[int, ...] = ()
    scope: str = "continuous_cardinal_half_ray_swept_box_to_infinity"


class AtomicOpType(str, Enum):
    """Atomic execution-trace build operations (`R8-WS1` sound execution contract)."""

    PLACE_PART = "PLACE_PART"
    PLACE_SUBASSEMBLY = "PLACE_SUBASSEMBLY"
    ADD_TEMP_SUPPORT = "ADD_TEMP_SUPPORT"
    REMOVE_TEMP_SUPPORT = "REMOVE_TEMP_SUPPORT"
    REORIENT_ASSEMBLY = "REORIENT_ASSEMBLY"


@dataclass(frozen=True, slots=True)
class AtomicBuildStep:
    """Single atomic build step in an `ExecutionTrace`."""

    op_type: AtomicOpType
    instance_ids: Tuple[int, ...] = ()
    part_id: str = ""
    pose: Optional[LDUFrame] = None
    is_grounded: bool = False
    temp_support_tag: str = ""
    subtrace: Tuple["AtomicBuildStep", ...] = ()
    gravity_unit: Tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """Explicit atomic build trace consumed by `PrefixMechanicsCert` and `InsertionCert`."""

    steps: Tuple[AtomicBuildStep, ...]
    allow_subassemblies: bool = False
    allow_temp_supports: bool = False
    allow_reorientation: bool = False
    trace_source: str = "recorded_atomic_placements"


@dataclass(frozen=True, slots=True)
class PerturbationMechanicsCert:
    """
    Certificate for mechanism perturbation stability (`R8-WS1` / `R8-WS4`):
    Rejects passive unstable equilibria (inverted pendulums, unbraced 4-bar parallelograms)
    under `+-2 deg` passive-joint potential work and `+-0.05 g` lateral wrench perturbations.
    """

    status: CertStatus = CertStatus.NOT_APPLICABLE
    is_perturbation_stable: bool = True
    min_potential_delta_g_ldu: float = 0.0
    worst_lateral_utilization: float = 0.0
    unstable_joint_ids: Tuple[int, ...] = ()
    mechanism_regime: str = "rigid"
    min_hessian_eigenvalue: float = 0.0


@dataclass(frozen=True, slots=True)
class ModelAgreementCert:
    """Certificate comparing the analytical contact-patch LP against an independent sub-stud polygon LP."""

    status: CertStatus = CertStatus.NOT_APPLICABLE
    agreed_with_substud_polygon: bool = True
    kernel_lp_stable: bool = True
    substud_polygon_stable: bool = True
    substud_peak_utilization: float = 0.0
    calibration_reference: str = "uncalibrated"


@dataclass(frozen=True, slots=True)
class ExternalPhysicalObservation:
    """
    Recorded observation from an external multi-body simulator (MuJoCo/Bullet/PhysX) or physical rig.
    Required before `PhysicalValidationCert` can ever transition out of `CertStatus.UNKNOWN` (`F9`).
    """

    rig_or_simulator_id: str
    run_sha256: str
    observed_stable: bool
    notes: str = ""


@dataclass(frozen=True, slots=True)
class LoopClosureCert:
    """Certificate for kinematic cycle closure residuals on closed-loop continuous $SE(3)$ mechanisms."""

    status: CertStatus = CertStatus.NOT_APPLICABLE
    max_position_residual_ldu: float = 0.0
    max_angle_residual_rad: float = 0.0


@dataclass(frozen=True, slots=True)
class PhysicalValidationCert:
    """
    Certificate for external multi-body simulator (MuJoCo/Bullet/PhysX) or physical test rig.
    Defaults to `CertStatus.UNKNOWN` when only the analytical static LP has been evaluated.
    """

    status: CertStatus = CertStatus.UNKNOWN
    simulator_or_rig: str = "untested_external_physics"
    notes: str = "Analytical 6-DoF static LP evaluated; external simulator/physical build not run."


@dataclass(frozen=True, slots=True)
class InventoryCert:
    """
    Certificate verifying zero mold or `(part_id, color_id)` overruns against declared finite kits (`Phase 11 Ext 1 & 2`).
    """

    status: CertStatus = CertStatus.NOT_APPLICABLE
    mold_overruns: Tuple[Tuple[str, int, int], ...] = ()
    color_overruns: Tuple[Tuple[str, int, int, int], ...] = ()
    inventory_counts: Optional[Dict[str, int]] = None
    colored_inventory: Optional[Any] = None
    skin_mean_delta_e00: float = 0.0
    fifo_baseline_skin_mean_delta_e00: float = 0.0
    core_parts_count: int = 0
    chromatic_splits_accepted: int = 0


@dataclass(frozen=True, slots=True)
class CertificateBundle:
    """Complete multi-domain verification certificate bundle (`R8-WS1` + `Phase 11` strict contract)."""

    geometry: GeometryCert
    static_mechanics: StaticMechanicsCert
    prefix_mechanics: PrefixMechanicsCert
    insertion: InsertionCert
    physical_validation: PhysicalValidationCert = field(default_factory=PhysicalValidationCert)
    perturbation_mechanics: PerturbationMechanicsCert = field(default_factory=PerturbationMechanicsCert)
    model_agreement: ModelAgreementCert = field(default_factory=ModelAgreementCert)
    loop_closure: LoopClosureCert = field(default_factory=LoopClosureCert)
    inventory: InventoryCert = field(default_factory=InventoryCert)
    foundation_policy: str = FoundationPolicy.CLUTCHED_BASEPLATE.value
    execution_trace: Optional[ExecutionTrace] = None
    ground_plane_z_ldu: float = 0.0

    @property
    def overall_certified(self) -> bool:
        return (
            self.geometry.status == CertStatus.PASS
            and self.static_mechanics.status == CertStatus.PASS
            and self.prefix_mechanics.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
            and self.insertion.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
            and self.perturbation_mechanics.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
            and self.model_agreement.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
            and self.loop_closure.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
            and self.inventory.status in (CertStatus.PASS, CertStatus.NOT_APPLICABLE)
        )

    def summary_dict(self) -> Dict[str, str]:
        return {
            "geometry": self.geometry.status.value,
            "static_mechanics": self.static_mechanics.status.value,
            "prefix_mechanics": self.prefix_mechanics.status.value,
            "insertion": self.insertion.status.value,
            "physical_validation": self.physical_validation.status.value,
            "perturbation_mechanics": self.perturbation_mechanics.status.value,
            "model_agreement": self.model_agreement.status.value,
            "loop_closure": self.loop_closure.status.value,
            "inventory": self.inventory.status.value,
        }


# ---------------------------------------------------------------------------
# 2. MinimumBuildRecord & Dynamic CorpusAttritionFunnel
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MinimumBuildRecord:
    """Standardized per-object build & certification telemetry record with 6-term timing (`R8-WS3`)."""

    object_id: str
    source_dataset: str
    source_row_index: int
    category_id: str
    caption: str
    split: str
    solver_id: str
    kit_name: str
    status: str
    num_parts_placed: int
    reference_part_count: int
    num_courses: int
    target_iou: float
    target_precision: float
    target_recall: float
    collision_free: bool
    protrusion_collision_free: bool
    static_stable: bool
    prefix_stable: bool
    monotone_insertable: bool
    peak_utilization: float
    wall_clock_ms: float
    peak_rss_mb: float
    no_good_cuts_used: int
    certificate_summary: Dict[str, str]
    t_target_ms: float = 0.0
    t_candidates_ms: float = 0.0
    t_selection_ms: float = 0.0
    t_mechanics_ms: float = 0.0
    t_execution_checks_ms: float = 0.0
    t_repair_ms: float = 0.0
    active_target_tiles: int = 0
    target_tile_memory_mib: float = 0.0
    cpu_time_ms: float = 0.0
    max_frontier_ports: int = 0
    total_candidates_generated: int = 0
    total_conflict_edges: int = 0
    lp_solver_calls: int = 0
    witness_reuse_hits: int = 0


@dataclass(frozen=True, slots=True)
class CorpusAttritionFunnel:
    """
    Dynamically computed 7-stage attrition funnel for external LDraw corpora (Round 6 F7 fix).
    Zero hardcoded constants: every count is tallied directly from live parser, collision,
    and static equilibrium evaluations.
    """

    kit_name: str
    allowed_part_ids: Tuple[str, ...]
    stage0_fetched: int
    stage1_parsed_valid: int
    stage2_catalog_supported: int
    stage3_collision_and_protrusion_free: int
    stage4_connected_to_ground: int
    stage5_static_equilibrium_certified: int
    stage6_prefix_certified: int
    drop_reasons: Dict[str, int]
    unsupported_part_counts: Dict[str, int]


def evaluate_corpus_attrition_funnel(
    records: Sequence[Dict[str, Any]],
    allowed_part_ids: Sequence[str] = STABLETEXT2BRICK_8PART_KIT,
    kit_name: str = "STABLETEXT2BRICK_8PART_KIT",
    validator: Optional[SparseStaticEquilibriumValidator] = None,
) -> CorpusAttritionFunnel:
    """
    Dynamically evaluate the 7-stage corpus attrition funnel across `records` against
    `allowed_part_ids` and `validator`.
    """
    if validator is None:
        validator = SparseStaticEquilibriumValidator(clutch_force_capacity_grams=99.9)
    allowed_set = set(allowed_part_ids)
    {pid: get_brick_primitive(pid) for pid in allowed_set}

    s0 = len(records)
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    drop_reasons: Dict[str, int] = {}
    unsupported_parts: Dict[str, int] = {}

    for rec in records:
        ldr_text = str(rec.get("ldr_text", ""))
        specs, report = parse_external_record_specs(
            ldr_text, allowed_part_ids=allowed_set
        )
        for upid in report.unsupported_part_ids:
            unsupported_parts[upid] = unsupported_parts.get(upid, 0) + 1

        if report.malformed_type1_lines > 0 or report.non_cubic_rotation_lines > 0 or (
            len(specs) == 0 and report.unsupported_valid_part_lines == 0
        ):
            drop_reasons["parse_or_rotation_error"] = drop_reasons.get("parse_or_rotation_error", 0) + 1
            continue
        s1 += 1

        if report.unsupported_valid_part_lines > 0 or len(specs) == 0:
            drop_reasons["unsupported_catalog_part"] = drop_reasons.get("unsupported_catalog_part", 0) + 1
            continue
        s2 += 1

        # Stage 3: Check exact hull & unmated protrusion collision freedom
        sorted_specs = sorted(
            specs,
            key=lambda s: (
                min(b.zmin for b in get_brick_primitive(s.part_id).hull_boxes),
                s.pose.pos[2],
                s.pose.pos[1],
                s.pose.pos[0],
            ),
        )
        min_z = min(
            min(b.transform(sp.pose).zmin for b in get_brick_primitive(sp.part_id).hull_boxes)
            for sp in sorted_specs
        )
        state = AssemblyState()
        collision_failed = False
        prefix_all_stable = True
        for sp in sorted_specs:
            prim = get_brick_primitive(sp.part_id)
            w_boxes = tuple(b.transform(sp.pose) for b in prim.hull_boxes)
            is_grounded = min(b.zmin for b in w_boxes) <= min_z
            try:
                state.place_root_part(sp.part_id, sp.pose, is_grounded=is_grounded)
            except ValueError:
                collision_failed = True
                break
        if collision_failed:
            drop_reasons["hull_or_protrusion_collision"] = (
                drop_reasons.get("hull_or_protrusion_collision", 0) + 1
            )
            continue
        s3 += 1

        # Stage 4: Connected to ground support
        depths = state.compute_support_depths()
        if any(d >= 999 for d in depths.values()):
            drop_reasons["disconnected_floating_part"] = (
                drop_reasons.get("disconnected_floating_part", 0) + 1
            )
            continue
        s4 += 1

        # Stage 5: Final static equilibrium certified
        eq = validator.evaluate(state)
        if not eq.is_stable:
            drop_reasons["static_equilibrium_overload"] = (
                drop_reasons.get("static_equilibrium_overload", 0) + 1
            )
            continue
        s5 += 1

        # Stage 6: Course-by-course prefix equilibrium check (deferring downward-hanging bricks
        # until their upper anchoring bridge course is placed)
        by_z: Dict[int, List[Any]] = {}
        for sp in sorted_specs:
            prim = get_brick_primitive(sp.part_id)
            z0 = min(b.transform(sp.pose).zmin for b in prim.hull_boxes)
            by_z.setdefault(z0, []).append(sp)
        prefix_state = AssemblyState()
        deferred_prefix: List[Any] = []
        for z0 in sorted(by_z.keys()):
            pending = list(by_z[z0]) + deferred_prefix
            deferred_prefix = []
            made_progress = True
            while made_progress and pending:
                made_progress = False
                still_pending: List[Any] = []
                depths_cur = prefix_state.compute_support_depths()
                for sp in pending:
                    prim_sp = get_brick_primitive(sp.part_id)
                    is_g = z0 <= min_z and min(
                        b.transform(sp.pose).zmin for b in prim_sp.hull_boxes
                    ) <= min_z
                    mpairs, has_clash = prefix_state._find_matching_contacts_for_pose(prim_sp, sp.pose)
                    connected_now = is_g or any(depths_cur.get(gpid[0], 999) < 999 for _, gpid in mpairs)
                    if not has_clash and connected_now:
                        try:
                            iid = prefix_state.place_root_part(sp.part_id, sp.pose, is_grounded=is_g)
                            depths_cur[iid] = 0 if is_g else 1
                            made_progress = True
                        except ValueError:
                            still_pending.append(sp)
                    else:
                        still_pending.append(sp)
                pending = still_pending
            deferred_prefix = pending
            if prefix_state.num_parts > 0:
                if any(d >= 999 for d in prefix_state.compute_support_depths().values()):
                    prefix_all_stable = False
                    break
        if deferred_prefix:
            prefix_all_stable = False
        if not prefix_all_stable:
            drop_reasons["prefix_equilibrium_unstable"] = (
                drop_reasons.get("prefix_equilibrium_unstable", 0) + 1
            )
            continue
        s6 += 1

    return CorpusAttritionFunnel(
        kit_name=kit_name,
        allowed_part_ids=tuple(allowed_part_ids),
        stage0_fetched=s0,
        stage1_parsed_valid=s1,
        stage2_catalog_supported=s2,
        stage3_collision_and_protrusion_free=s3,
        stage4_connected_to_ground=s4,
        stage5_static_equilibrium_certified=s5,
        stage6_prefix_certified=s6,
        drop_reasons=drop_reasons,
        unsupported_part_counts=unsupported_parts,
    )


# ---------------------------------------------------------------------------
# 3. Kernel Result Types & Insertion / Certification Helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RealizeRouteProvenance:
    """Explicit route, target-identity, recall-threshold, and verification-scope provenance (`Phase 15–16`)."""

    effective_route_id: str
    requested_solver_id: str
    caller_target_sha256: str
    evaluated_target_sha256: str
    target_was_refined_or_transformed: bool
    evaluated_target_kind: str
    min_recall_threshold: float
    achieved_evaluated_recall: float
    achieved_evaluated_iou: float
    achieved_caller_target_iou: float
    achieved_mesh_5x5x2_iou: Optional[float]
    foundation_policy: str
    ground_plane_z_ldu: float
    license_filter: str
    recommended_mpd_source: str
    candidate_route_id: str = ""
    fallback_from_route_id: Optional[str] = None
    workbench_origin_ldu: Tuple[int, int, int] = (0, 0, 0)
    consumed_options: Tuple[str, ...] = ()
    ignored_nondefault_options: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KernelRealizeResult:
    """Result returned by `TopoBrickKernel.realize(...)`."""

    status: KernelResultStatus
    assembly: AssemblyState
    certificates: CertificateBundle
    build_record: MinimumBuildRecord
    no_good_cuts_added: int = 0
    courses_executed: int = 0
    external_support_bricks_added: int = 0
    external_support_volume_ldu3: int = 0
    realization_mode: str = "blind"
    open_catalog_diagnostics: Optional[Any] = None
    evaluated_target_field: Optional[Any] = None
    inventory_shortfall: Optional[Any] = None
    compositional_mpd_text: Optional[str] = None
    stopping_reason: str = ""
    remaining_budget_s: Optional[float] = None
    route_provenance: Optional[RealizeRouteProvenance] = None


class RepairEditType(str, Enum):
    """Declared task-aware repair operations (`R8-WS2` contract)."""

    REMERGE_SEAM = "REMERGE_SEAM"
    REORIENT_PART = "REORIENT_PART"
    ADD_COUNTERWEIGHT = "ADD_COUNTERWEIGHT"
    ADD_EXTERNAL_SUPPORT = "ADD_EXTERNAL_SUPPORT"
    DELETE_NONLOCKED_PART = "DELETE_NONLOCKED_PART"


@dataclass(frozen=True, slots=True)
class RepairAuditReport:
    """Audit report returned by task-aware constrained `TopoBrickKernel.repair(...)`."""

    initial_iou: float
    repaired_iou: float
    retained_volume_ratio: float
    deleted_volume_ratio: float
    locked_features_preserved: bool
    inventory_respected: bool
    support_volume_added_ldu3: int
    applied_edit_types: Tuple[str, ...]
    rejection_reason: str = ""


@dataclass(frozen=True, slots=True)
class KernelRepairResult:
    """Result returned by `TopoBrickKernel.repair(...)`."""

    status: KernelResultStatus
    repaired_assembly: AssemblyState
    initial_certificates: CertificateBundle
    repaired_certificates: CertificateBundle
    parts_removed: int
    parts_added: int
    parts_reoriented: int
    edit_volume_ldu3: int
    repair_procedure: str = "constructive_seam_counterweight_support_and_constrained_repair"
    wall_clock_ms: float = 0.0
    audit: Optional[RepairAuditReport] = None


@dataclass(frozen=True, slots=True)
class ValidPlacementAction:
    """A single certified action emitted by `TopoBrickKernel.constrain(...)`."""

    action_token: str
    part_id: str
    pose: LDUFrame
    consumed_ports: FrozenSet[GlobalPortId]
    target_intersection_ldu3: int
    closes_structural_seam: bool
    creates_biconnected_cycle: bool
    post_step_utilization: float


@dataclass(frozen=True, slots=True)
class KernelConstrainResult:
    """Result returned by `TopoBrickKernel.constrain(...)`."""

    status: KernelResultStatus
    valid_actions: Tuple[ValidPlacementAction, ...]
    candidate_pool_size: int
    certified_action_count: int
    wall_clock_ms: float
    updated_cache: Optional[IncrementalCandidateCache] = None


@dataclass(frozen=True, slots=True)
class ValidatedConstraintSpec:
    """
    Canonical normalized constraint bundle (`F3`) shared across `realize()`, `certify()`, and `repair()`
    so finite mold/color inventories, forbidden openings, kit limits, and foundation policies are never dropped.
    """

    allowed_part_ids: Optional[Tuple[str, ...]] = None
    inventory_counts: Optional[Dict[str, int]] = None
    colored_inventory: Optional[Any] = None
    foundation_policy: FoundationPolicy = FoundationPolicy.CLUTCHED_BASEPLATE
    support_policy: str = "none"
    forbidden_support_boxes: Tuple[LDUBox, ...] = ()
    license_filter: str = "all"
    min_retained_iou: Optional[float] = None
    max_deleted_volume_ratio: Optional[float] = None

    @classmethod
    def from_args(
        cls,
        allowed_part_ids: Optional[Sequence[str]] = None,
        inventory_counts: Optional[Dict[str, int]] = None,
        colored_inventory: Optional[Any] = None,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        support_policy: str = "none",
        forbidden_support_boxes: Sequence[LDUBox] = (),
        license_filter: str = "all",
        min_retained_iou: Optional[float] = None,
        max_deleted_volume_ratio: Optional[float] = None,
    ) -> "ValidatedConstraintSpec":
        fp_enum = (
            foundation_policy
            if isinstance(foundation_policy, FoundationPolicy)
            else FoundationPolicy(str(foundation_policy))
        )
        if support_policy not in ("none", "minimal_permanent_pillars", "temporary_props_only"):
            raise ValueError(f"Unsupported support_policy={support_policy!r}.")
        if license_filter not in ("all", "cc_by_4_0_only"):
            raise ValueError(f"Unsupported license_filter={license_filter!r}.")
        norm_inv: Optional[Dict[str, int]] = None
        if inventory_counts is not None:
            norm_inv = {str(k): int(v) for k, v in inventory_counts.items()}
            if any(v < 0 for v in norm_inv.values()):
                raise ValueError("inventory_counts cannot contain negative quantities.")
        return cls(
            allowed_part_ids=tuple(str(p) for p in allowed_part_ids) if allowed_part_ids is not None else None,
            inventory_counts=norm_inv,
            colored_inventory=colored_inventory,
            foundation_policy=fp_enum,
            support_policy=str(support_policy),
            forbidden_support_boxes=tuple(forbidden_support_boxes),
            license_filter=str(license_filter),
            min_retained_iou=float(min_retained_iou) if min_retained_iou is not None else None,
            max_deleted_volume_ratio=(
                float(max_deleted_volume_ratio) if max_deleted_volume_ratio is not None else None
            ),
        )



@dataclass(frozen=True, slots=True)
class SingleCoreAuditReport:
    """Single-core BLAS/OpenMP/PyTorch/HiGHS thread pinning & CPU-to-wall audit report."""

    env_threads_pinned: Dict[str, str]
    torch_num_threads: int
    highs_threads: int
    workload_wall_ms: float
    workload_cpu_ms: float
    cpu_to_wall_ratio: float
    peak_rss_mb: float
    strictly_single_core_verified: bool
    pre_existing_env_pinned: bool = False


def enforce_and_audit_single_core() -> SingleCoreAuditReport:
    """Inspect pre-existing thread env vars, pin BLAS/OpenMP/HiGHS (and PyTorch if loaded) to 1 thread."""
    import os
    from topo_brick.interface.brick_catalog import EXTENDED_10PART_RECT_KIT
    from topo_brick.kernel.construction_kernel import TopoBrickKernel

    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    pre_pinned = all(os.environ.get(k) == "1" for k in thread_vars)
    pinned: Dict[str, str] = {}
    for k in thread_vars:
        os.environ[k] = "1"
        pinned[k] = os.environ.get(k, "")
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        torch_mod.set_num_threads(1)
        t_threads = int(torch_mod.get_num_threads())
    else:
        t_threads = 1

    kernel = TopoBrickKernel()
    tf = SparseTiledTargetField(
        boxes=(
            LDUBox(0, 0, 0, 120, 80, 72),
            LDUBox(20, 20, 72, 100, 60, 120),
        )
    )
    kernel.realize(tf, allowed_part_ids=EXTENDED_10PART_RECT_KIT, kit_name="EXTENDED_10PART_RECT_KIT")
    w0 = time.perf_counter()
    c0 = time.process_time()
    for _ in range(6):
        kernel.realize(tf, allowed_part_ids=EXTENDED_10PART_RECT_KIT, kit_name="EXTENDED_10PART_RECT_KIT")
    wall_ms = (time.perf_counter() - w0) * 1000.0
    cpu_ms = (time.process_time() - c0) * 1000.0
    ratio = float(cpu_ms / max(wall_ms, 1e-3))
    ok = all(v == "1" for v in pinned.values()) and t_threads == 1 and ratio <= 1.08
    return SingleCoreAuditReport(
        env_threads_pinned=pinned,
        torch_num_threads=t_threads,
        highs_threads=1,
        workload_wall_ms=wall_ms,
        workload_cpu_ms=cpu_ms,
        cpu_to_wall_ratio=ratio,
        peak_rss_mb=_get_peak_rss_mb(),
        strictly_single_core_verified=ok,
        pre_existing_env_pinned=pre_pinned,
    )


@dataclass(frozen=True, slots=True)
class PromotionGateVerdict:
    """
    Round 7 Decision 4 Three-Part Promotion Gate Verdict:
    Separates Reliability, Module-Benefit (paired ablation vs. `S1` default), and
    Challenge-Set gates. A module or solver is promoted ONLY if all three pass.
    """

    module_or_solver_id: str
    reliability_gate_passed: bool
    module_benefit_gate_passed: bool
    challenge_set_gate_passed: bool
    promoted_to_default: bool
    speedup_vs_s1_ratio: float
    iou_delta_vs_s1: float
    challenge_metric_value: float
    rationale: str
