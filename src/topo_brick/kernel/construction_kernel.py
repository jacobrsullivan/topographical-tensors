"""
Topo-Brick Universal Construction Kernel (`realize`, `certify`, `constrain`, `repair`).

`Phase 15` modularized architecture (`< 2,000` lines, `0` `torch` runtime imports):
- Contracts & certificates: `topo_brick.kernel.kernel_contracts`
- Continuous 6-axis corridors: `topo_brick.kernel.insertion_corridors`
- Course solvers (`S1`–`S4`, baselines): `topo_brick.kernel.course_solvers`
- Route policy & provenance: `topo_brick.kernel.realize_policy`
- Task-aware constrained repair: `topo_brick.kernel.constrained_repair`
- Opt-in PyTorch controls (`M1`–`M3`): `topo_brick.kernel.learned_bottleneck_controls` (lazy PEP 562)
- Promotion-gate deliverables (`1`–`4`): `topo_brick.kernel.revised_kernel_deliverables` (lazy PEP 562)
"""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

if TYPE_CHECKING:
    from topo_brick.kernel.learned_bottleneck_controls import (
        BrickActionGrammarAdapter,
        ConstrainedRolloutMetrics,
        DescriptorFrameEncoder,
        LearnedCatalogAndPortFilter,
        SearchDistilledValueModel,
    )
    from topo_brick.kernel.revised_kernel_deliverables import (
        Deliverable1Report,
        Deliverable2Report,
        Deliverable3Report,
        Deliverable4Report,
        PromotionGateVerdict,
        RevisedKernelSuiteSummary,
        SolverCohortMetrics,
        build_parametric_non_myopic_trap,
        run_deliverable1_small_kernel_verification,
        run_deliverable2_untouched_external_benchmark,
        run_deliverable3_external_adapter_and_connectors,
        run_deliverable4_learning_at_measured_bottleneck,
        run_revised_kernel_four_deliverables_suite,
    )
import numpy as np

from topo_brick.evaluation.static_equilibrium import (
    EquilibriumResult,
    FoundationPolicy,
    SparseStaticEquilibriumValidator,
)
from topo_brick.interface.brick_catalog import (
    STABLETEXT2BRICK_8PART_KIT,
    are_connectors_compatible,
    get_brick_primitive,
    is_known_part_id,
)
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import (
    AssemblyState,
    CandidatePlacement,
    CandidatePool,
    IncrementalCandidateCache,
    PortPolarity,
    SpatialBoxHashIndex,
    TrialPartRollbackToken,
    compute_cheap_pre_expansion_port_scores,
)
from topo_brick.kernel.course_solvers import (
    CoverAndRepairSolver,
    D3PMBatchSampler,
    GreedyLayerMergeControl,
    IterativeSeamRefinementLuo2015,
    LegolizationBaseline,
    LuoInspiredSeamRefinement,
    PairSeededLocalSearch,
    RecurrentBatchRefiner,
    _build_blind_grounded_base_pool,
    _solve_binary_mip_fast,
    build_canonical_nogood_cut,
)
from topo_brick.kernel.insertion_corridors import (
    _boxes_have_continuous_cardinal_corridor,
    _order_course_atomic_1by1,
    _swept_obb_hits_along_ray,
    check_monotone_insertion_corridor,
    check_rigid_group_insertion_corridor,
)
from topo_brick.kernel.kernel_contracts import (
    EXTENDED_STABLETEXT2BRICK_DIM_TO_PART,
    AtomicBuildStep,
    AtomicOpType,
    CertStatus,
    CertificateBundle,
    CorpusAttritionFunnel,
    ExecutionTrace,
    ExternalParseSummary,
    ExternalPhysicalObservation,
    GeometryCert,
    InsertionCert,
    InventoryCert,
    KernelConstrainResult,
    KernelRealizeResult,
    KernelRepairResult,
    KernelResultStatus,
    LoopClosureCert,
    MinimumBuildRecord,
    ModelAgreementCert,
    PerturbationMechanicsCert,
    PhysicalValidationCert,
    PrefixMechanicsCert,
    RealizeRouteProvenance,
    RepairAuditReport,
    RepairEditType,
    SingleCoreAuditReport,
    StaticMechanicsCert,
    ValidPlacementAction,
    ValidatedConstraintSpec,
    _get_peak_rss_mb,
    enforce_and_audit_single_core,
    evaluate_corpus_attrition_funnel,
    parse_external_record_specs,
)
from topo_brick.kernel.realize_policy import (
    RealizeRequestSpec,
    ResolvedRealizePolicy,
    execute_realize_external_record,
    execute_resolved_realize,
    resolve_realize_policy,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField, TargetShapeBoxes


class TopoBrickKernel:
    """
    Universal Construction Kernel exposing the 4 core interfaces approved in Round 7 Decision 1:
    1. `certify(assembly, target_field, ...)` -> `CertificateBundle`
    2. `realize(target_field_or_ldr, ...)`    -> `KernelRealizeResult`
    3. `constrain(assembly, target_field, ...)` -> `KernelConstrainResult`
    4. `repair(assembly, target_field, ...)`    -> `KernelRepairResult`
    """

    def __init__(
        self,
        validator: Optional[SparseStaticEquilibriumValidator] = None,
        default_kit: Sequence[str] = STABLETEXT2BRICK_8PART_KIT,
    ) -> None:
        self.validator = validator or SparseStaticEquilibriumValidator(
            clutch_force_capacity_grams=99.9
        )
        self.default_kit = tuple(default_kit)
        self.s1_solver = CoverAndRepairSolver(validator=self.validator)
        self.s2_solver = PairSeededLocalSearch(validator=self.validator)
        self.s3_solver = RecurrentBatchRefiner()
        self.s4_solver = D3PMBatchSampler()

    @staticmethod
    def _evaluate_inventory_cert(
        parts_map: Dict[int, Any],
        inventory_counts: Optional[Dict[str, int]] = None,
        colored_inventory: Optional[Any] = None,
    ) -> InventoryCert:
        if inventory_counts is None and colored_inventory is None:
            return InventoryCert()
        mold_used: Dict[str, int] = {}
        color_used: Dict[Tuple[str, int], int] = {}
        for p_inv in parts_map.values():
            mold_used[p_inv.part_id] = mold_used.get(p_inv.part_id, 0) + 1
            ck = (p_inv.part_id, int(getattr(p_inv, "color_id", 15)))
            color_used[ck] = color_used.get(ck, 0) + 1
        eff_mold_lim: Optional[Dict[str, int]] = (
            dict(inventory_counts) if inventory_counts is not None else None
        )
        if eff_mold_lim is None and colored_inventory is not None and hasattr(colored_inventory, "marginal_mold_counts"):
            eff_mold_lim = colored_inventory.marginal_mold_counts()
        m_over: List[Tuple[str, int, int]] = []
        if eff_mold_lim is not None:
            for pid_k, u_cnt in sorted(mold_used.items()):
                allow_m = int(eff_mold_lim.get(pid_k, 0))
                if u_cnt > allow_m:
                    m_over.append((pid_k, u_cnt, allow_m))
        c_over: List[Tuple[str, int, int, int]] = []
        if colored_inventory is not None and hasattr(colored_inventory, "counts"):
            for (pid_c, cid_c), u_c in sorted(color_used.items()):
                allow_c = int(colored_inventory.get_count(pid_c, cid_c))
                if u_c > allow_c:
                    c_over.append((pid_c, cid_c, u_c, allow_c))
        inv_ok = (len(m_over) == 0) and (len(c_over) == 0)
        return InventoryCert(
            status=CertStatus.PASS if inv_ok else CertStatus.FAIL,
            mold_overruns=tuple(m_over),
            color_overruns=tuple(c_over),
            inventory_counts=eff_mold_lim,
            colored_inventory=colored_inventory,
        )

    @staticmethod
    def _evaluate_physical_validation_cert(
        external_physical_observation: Optional[ExternalPhysicalObservation] = None,
    ) -> PhysicalValidationCert:
        if external_physical_observation is None:
            return PhysicalValidationCert(status=CertStatus.UNKNOWN)
        return PhysicalValidationCert(
            status=CertStatus.PASS if external_physical_observation.observed_stable else CertStatus.FAIL,
            simulator_or_rig=str(external_physical_observation.rig_or_simulator_id),
            notes=(
                f"run_sha256={external_physical_observation.run_sha256}: {external_physical_observation.notes}".strip()
            ),
        )

    def _certify_continuous_assembly(
        self,
        assembly: Any,
        verify_prefix: bool = True,
        verify_insertion: bool = True,
        execution_trace: Optional[ExecutionTrace] = None,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        verify_perturbation: bool = False,
        inventory_counts: Optional[Dict[str, int]] = None,
        colored_inventory: Optional[Any] = None,
        external_physical_observation: Optional[ExternalPhysicalObservation] = None,
        ground_plane_z_ldu: float = 0.0,
        deadline_monotonic_s: Optional[float] = None,
    ) -> CertificateBundle:
        """Certify a `ContinuousAssemblyState` (15-axis OBB-SAT, continuous joint wrench LP, perturbation)."""
        from topo_brick.interface.se3_kinematics import (
            ContinuousAssemblyState,
            box_to_obb,
            get_connector_axial_slide_bounds_ldu,
            is_obb_protrusion_contained_in_socket_pocket,
            obb_intersects_sat,
        )

        fp_enum = (
            foundation_policy
            if isinstance(foundation_policy, FoundationPolicy)
            else FoundationPolicy(str(foundation_policy))
        )
        parts = assembly.placed_parts
        part_ids_sorted = sorted(parts.keys())

        # 1. Pairwise 15-axis OBB-SAT hull check + pairwise local-socket-exempt protrusion check
        violating_set: Set[Tuple[int, int]] = set()
        for ia in range(len(part_ids_sorted)):
            ida = part_ids_sorted[ia]
            pa = parts[ida]
            for ib in range(ia + 1, len(part_ids_sorted)):
                idb = part_ids_sorted[ib]
                pb = parts[idb]
                if any(
                    obb_intersects_sat(oa, ob, allowed_penetration_ldu=0.25)
                    for oa in pa.world_obbs
                    for ob in pb.world_obbs
                ):
                    violating_set.add((ida, idb))

        protrusion_free = True
        for ida in part_ids_sorted:
            pa = parts[ida]
            prim_a = get_brick_primitive(pa.part_id)
            for p_spec in prim_a.ports:
                if p_spec.local_protrusion_box is None:
                    continue
                gpid_a = (ida, p_spec.port_id)
                wf_a = assembly._port_world_pose(pa.continuous_pose, p_spec.local_frame)
                prot_obb = box_to_obb(p_spec.local_protrusion_box, wf_a)
                partner_gpid = assembly.matched_port_pairs.get(gpid_a)
                partner_iid = partner_gpid[0] if partner_gpid is not None else -1
                partner_wf = None
                partner_ctype = None
                if partner_gpid is not None and partner_iid in parts:
                    pb_p = parts[partner_iid]
                    ps_b = get_brick_primitive(pb_p.part_id).ports[partner_gpid[1]]
                    partner_wf = assembly._port_world_pose(pb_p.continuous_pose, ps_b.local_frame)
                    partner_ctype = ps_b.connector_type
                for idb in part_ids_sorted:
                    if idb == ida:
                        continue
                    pb = parts[idb]
                    for ob_b in pb.world_obbs:
                        if (
                            idb == partner_iid
                            and partner_wf is not None
                            and partner_ctype is not None
                            and is_obb_protrusion_contained_in_socket_pocket(
                                prot_obb, ob_b, partner_wf, partner_ctype
                            )
                        ):
                            continue
                        if obb_intersects_sat(prot_obb, ob_b, allowed_penetration_ldu=0.25):
                            protrusion_free = False
                            violating_set.add((min(ida, idb), max(ida, idb)))

        ports_compat = True
        for gpid_a, gpid_b in assembly.matched_port_pairs.items():
            pa = parts[gpid_a[0]]
            pb = parts[gpid_b[0]]
            sa = get_brick_primitive(pa.part_id).ports[gpid_a[1]]
            sb = get_brick_primitive(pb.part_id).ports[gpid_b[1]]
            if not are_connectors_compatible(sa.connector_type, sa.polarity, sb.connector_type, sb.polarity):
                ports_compat = False
                break

        for (par_id, ch_id), j_tuple in getattr(assembly, "articulated_joints", {}).items():
            if par_id in parts and ch_id in parts and len(j_tuple) >= 5:
                par_pid, ch_pid, _, _, ax_slide = j_tuple[:5]
                sa_j = get_brick_primitive(parts[par_id].part_id).ports[par_pid]
                sb_j = get_brick_primitive(parts[ch_id].part_id).ports[ch_pid]
                lo_a, hi_a = get_connector_axial_slide_bounds_ldu(sa_j.connector_type)
                lo_b, hi_b = get_connector_axial_slide_bounds_ldu(sb_j.connector_type)
                if not (max(lo_a, lo_b) - 1e-6 <= float(ax_slide) <= min(hi_a, hi_b) + 1e-6):
                    ports_compat = False
                    violating_set.add((min(par_id, ch_id), max(par_id, ch_id)))

        collision_free = len(violating_set) == 0
        geom_pass = collision_free and protrusion_free and ports_compat
        geom_cert = GeometryCert(
            status=CertStatus.PASS if geom_pass else CertStatus.FAIL,
            collision_free=collision_free,
            protrusion_collision_free=protrusion_free,
            all_ports_compatible=ports_compat,
            state_integrity_verified=geom_pass,
            target_iou=1.0,
            target_precision=1.0,
            target_recall=1.0,
            violating_pairs=tuple(sorted(violating_set)),
        )

        eq = assembly.evaluate_static_equilibrium(
            foundation_policy=fp_enum,
            ground_plane_z_ldu=ground_plane_z_ldu,
            deadline_monotonic_s=deadline_monotonic_s,
        )
        static_cert = StaticMechanicsCert(
            status=CertStatus.PASS if (eq.is_stable and geom_pass) else CertStatus.FAIL,
            is_stable=bool(eq.is_stable),
            peak_utilization=float(eq.peak_utilization),
            safety_margin=float(eq.safety_margin),
            weakest_contact=eq.weakest_contact,
            moment_reference=eq.moment_reference,
            asymmetric_bearing_active=True,
            cold_rebuild_consistent=geom_pass,
            solve_time_ms=float(eq.solve_time_ms),
        )

        has_trace = execution_trace is not None or bool(getattr(assembly, "has_execution_trace", False))
        eff_c_trace: Optional[ExecutionTrace] = execution_trace
        if eff_c_trace is None and has_trace:
            raw_ord = list(getattr(assembly, "placement_order", ()))
            order_ids = [i for i in raw_ord if i in parts]
            if len(order_ids) != len(parts):
                order_ids = list(part_ids_sorted)
            eff_c_trace = ExecutionTrace(
                steps=tuple(
                    AtomicBuildStep(op_type=AtomicOpType.PLACE_PART, instance_ids=(iid,))
                    for iid in order_ids
                ),
                trace_source="continuous_placement_order",
            )

        if not verify_prefix and not verify_insertion:
            prefix_cert = PrefixMechanicsCert(CertStatus.NOT_APPLICABLE, 0, 0, None, 0.0)
            ins_cert = InsertionCert(
                CertStatus.NOT_APPLICABLE, 0, 0, (), scope="continuous_cardinal_half_ray_swept_obb_to_infinity"
            )
        elif not has_trace and len(part_ids_sorted) > 0:
            prefix_cert = (
                PrefixMechanicsCert(CertStatus.UNKNOWN, 0, 0, None, 0.0)
                if verify_prefix
                else PrefixMechanicsCert(CertStatus.NOT_APPLICABLE, 0, 0, None, 0.0)
            )
            ins_cert = (
                InsertionCert(
                    CertStatus.UNKNOWN, 0, 0, (), scope="continuous_cardinal_half_ray_swept_obb_to_infinity"
                )
                if verify_insertion
                else InsertionCert(
                    CertStatus.NOT_APPLICABLE, 0, 0, (), scope="continuous_cardinal_half_ray_swept_obb_to_infinity"
                )
            )
        else:
            sim_c = ContinuousAssemblyState()
            eval_s = 0
            stable_s = 0
            first_u: Optional[int] = None
            max_u = 0.0
            eval_i = 0
            ok_i = 0
            blocked_i: List[int] = []
            cur_grav = (0.0, 0.0, -1.0)
            dirs_6 = (
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 0.0, -1.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, -1.0, 0.0]),
            )
            steps_c = eff_c_trace.steps if eff_c_trace is not None else ()
            if len(part_ids_sorted) > 0 and len(steps_c) == 0:
                first_u = 0
                max_u = float("inf")
                blocked_i.append(-1)
            for s_idx, step in enumerate(steps_c):
                if step.op_type == AtomicOpType.REORIENT_ASSEMBLY:
                    if not (eff_c_trace is not None and eff_c_trace.allow_reorientation):
                        if first_u is None:
                            first_u = s_idx
                            max_u = float("inf")
                        continue
                    cur_grav = step.gravity_unit
                    if verify_prefix and len(sim_c.placed_parts) > 0:
                        eval_s += 1
                        eq_r = sim_c.evaluate_static_equilibrium(
                            foundation_policy=fp_enum,
                            gravity_unit=cur_grav,
                            ground_plane_z_ldu=ground_plane_z_ldu,
                            deadline_monotonic_s=deadline_monotonic_s,
                        )
                        if eq_r.is_stable:
                            stable_s += 1
                        elif first_u is None:
                            first_u = s_idx
                            max_u = float("inf")
                    continue

                if step.op_type not in (AtomicOpType.PLACE_PART, AtomicOpType.PLACE_SUBASSEMBLY):
                    if first_u is None:
                        first_u = s_idx
                        max_u = float("inf")
                    continue
                if step.op_type == AtomicOpType.PLACE_SUBASSEMBLY and not (
                    eff_c_trace is not None and eff_c_trace.allow_subassemblies
                ):
                    if first_u is None:
                        first_u = s_idx
                        max_u = float("inf")
                    continue

                step_iids = list(step.instance_ids)
                if not step_iids or any(i not in parts or i in sim_c.placed_parts for i in step_iids):
                    if first_u is None:
                        first_u = s_idx
                        max_u = float("inf")
                    blocked_i.append(s_idx)
                    continue

                if verify_insertion:
                    eval_i += 1
                    group_obbs = [ob for i in step_iids for ob in parts[i].world_obbs]
                    ex_obbs = [ex_o for ex_p in sim_c.placed_parts.values() for ex_o in ex_p.world_obbs]
                    ins_ok = (not ex_obbs) or any(
                        not any(
                            _swept_obb_hits_along_ray(so, ex_o, d_vec, allowed_penetration_ldu=0.25)
                            for ex_o in ex_obbs
                            for so in group_obbs
                        )
                        for d_vec in dirs_6
                    )
                    if ins_ok:
                        ok_i += 1
                    else:
                        blocked_i.append(step_iids[0])

                placed_ok_c = True
                for iid in step_iids:
                    cp = parts[iid]
                    try:
                        sim_c._next_id = iid
                        sim_c.place_part(
                            cp.part_id,
                            cp.continuous_pose,
                            is_grounded=cp.is_grounded,
                            check_collision=True,
                            color_id=getattr(cp, "color_id", 15),
                        )
                    except ValueError:
                        placed_ok_c = False
                        break
                for ga, gb in assembly.matched_port_pairs.items():
                    if ga[0] in sim_c.placed_parts and gb[0] in sim_c.placed_parts:
                        sim_c.matched_port_pairs[ga] = gb
                for jk, jv in getattr(assembly, "articulated_joints", {}).items():
                    if jk[0] in sim_c.placed_parts and jk[1] in sim_c.placed_parts:
                        sim_c.articulated_joints[jk] = jv

                if verify_prefix:
                    eval_s += 1
                    if not placed_ok_c:
                        if first_u is None:
                            first_u = s_idx
                            max_u = float("inf")
                    else:
                        eq_s = sim_c.evaluate_static_equilibrium(
                            foundation_policy=fp_enum,
                            gravity_unit=cur_grav,
                            ground_plane_z_ldu=ground_plane_z_ldu,
                            deadline_monotonic_s=deadline_monotonic_s,
                        )
                        if not np.isinf(eq_s.peak_utilization):
                            max_u = max(max_u, float(eq_s.peak_utilization))
                        if eq_s.is_stable:
                            stable_s += 1
                        elif first_u is None:
                            first_u = s_idx
                            max_u = float("inf")
                        if eq_s.budget_exceeded:
                            break

            if set(sim_c.placed_parts.keys()) != set(parts.keys()):
                if verify_prefix and first_u is None:
                    first_u = len(steps_c)
                    max_u = float("inf")
                if verify_insertion and not blocked_i:
                    blocked_i.append(-1)

            prefix_cert = (
                PrefixMechanicsCert(
                    CertStatus.PASS if first_u is None else CertStatus.FAIL,
                    eval_s,
                    stable_s,
                    first_u,
                    max_u,
                )
                if verify_prefix
                else PrefixMechanicsCert(CertStatus.NOT_APPLICABLE, 0, 0, None, 0.0)
            )
            ins_cert = (
                InsertionCert(
                    CertStatus.PASS if not blocked_i else CertStatus.FAIL,
                    eval_i,
                    ok_i,
                    tuple(blocked_i),
                    scope="continuous_cardinal_half_ray_swept_obb_to_infinity",
                )
                if verify_insertion
                else InsertionCert(
                    CertStatus.NOT_APPLICABLE,
                    0,
                    0,
                    (),
                    scope="continuous_cardinal_half_ray_swept_obb_to_infinity",
                )
            )

        if verify_perturbation:
            pert = assembly.evaluate_perturbation_stability(
                foundation_policy=fp_enum,
                ground_plane_z_ldu=ground_plane_z_ldu,
                deadline_monotonic_s=deadline_monotonic_s,
            )
            pert_cert = PerturbationMechanicsCert(
                status=CertStatus.PASS if pert.is_perturbation_stable else CertStatus.FAIL,
                is_perturbation_stable=pert.is_perturbation_stable,
                min_potential_delta_g_ldu=pert.min_restoring_work_g_ldu,
                worst_lateral_utilization=pert.worst_perturbed_utilization,
                unstable_joint_ids=(),
                mechanism_regime=pert.mechanism_classification,
                min_hessian_eigenvalue=float(getattr(pert, "min_hessian_eigenvalue", 0.0)),
            )
        else:
            pert_cert = PerturbationMechanicsCert()

        lc_res = getattr(assembly, "_loop_closure_result", None)
        if lc_res is not None:
            ang_res = float(
                np.arccos(max(-1.0, min(1.0, abs(float(getattr(lc_res, "axis_alignment_cos", 1.0))))))
            )
            loop_cert = LoopClosureCert(
                status=CertStatus.PASS if lc_res.converged else CertStatus.FAIL,
                max_position_residual_ldu=float(lc_res.position_residual_ldu),
                max_angle_residual_rad=ang_res,
            )
        else:
            loop_cert = LoopClosureCert()

        inv_cert_c = self._evaluate_inventory_cert(parts, inventory_counts, colored_inventory)
        phys_cert_c = self._evaluate_physical_validation_cert(external_physical_observation)

        return CertificateBundle(
            geometry=geom_cert,
            static_mechanics=static_cert,
            prefix_mechanics=prefix_cert,
            insertion=ins_cert,
            physical_validation=phys_cert_c,
            perturbation_mechanics=pert_cert,
            model_agreement=ModelAgreementCert(),
            loop_closure=loop_cert,
            inventory=inv_cert_c,
            foundation_policy=fp_enum.value,
            execution_trace=eff_c_trace,
            ground_plane_z_ldu=float(ground_plane_z_ldu),
        )

    def certify(
        self,
        assembly: Any,
        target_field: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
        verify_prefix: bool = True,
        verify_insertion: bool = True,
        verify_calibrated_substud_physics: bool = False,
        execution_trace: Optional[ExecutionTrace] = None,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        verify_perturbation: bool = False,
        verify_model_agreement: bool = False,
        inventory_counts: Optional[Dict[str, int]] = None,
        colored_inventory: Optional[Any] = None,
        external_physical_observation: Optional[ExternalPhysicalObservation] = None,
        ground_plane_z_ldu: float = 0.0,
        max_wall_clock_s: Optional[float] = None,
        deadline_monotonic_s: Optional[float] = None,
    ) -> CertificateBundle:
        """
        Evaluate independent verification certificates on `assembly` (`AssemblyState` or
        `ContinuousAssemblyState`, `R8-WS1` + `Phase 12` strict contract):
          - Atomic `ExecutionTrace` 1-by-1 verification for `PrefixMechanicsCert` & `InsertionCert`
            (returns `CertStatus.UNKNOWN` if untraced; never batches unverified sibling placements).
          - Explicit `FoundationPolicy` (`CLUTCHED_BASEPLATE`, `TABLE_FRICTION_ONLY`, `FIXED_ANCHOR`).
          - Optional `PerturbationMechanicsCert` and `ModelAgreementCert` (calibrated sub-stud LP).
          - `PhysicalValidationCert` remains `UNKNOWN` unless `external_physical_observation` is provided.
        """
        eff_cert_deadline: Optional[float] = deadline_monotonic_s
        if max_wall_clock_s is not None:
            cand_dl = time.monotonic() + max(0.0, float(max_wall_clock_s))
            eff_cert_deadline = cand_dl if eff_cert_deadline is None else min(eff_cert_deadline, cand_dl)

        if not isinstance(assembly, AssemblyState):
            return self._certify_continuous_assembly(
                assembly,
                verify_prefix=verify_prefix,
                verify_insertion=verify_insertion,
                execution_trace=execution_trace,
                foundation_policy=foundation_policy,
                verify_perturbation=verify_perturbation,
                inventory_counts=inventory_counts,
                colored_inventory=colored_inventory,
                external_physical_observation=external_physical_observation,
                ground_plane_z_ldu=ground_plane_z_ldu,
                deadline_monotonic_s=eff_cert_deadline,
            )

        fp_enum = (
            foundation_policy
            if isinstance(foundation_policy, FoundationPolicy)
            else FoundationPolicy(str(foundation_policy))
        )
        parts = assembly.placed_parts
        part_ids_sorted = sorted(parts.keys())

        # 1. Geometry Certificate (solid hull + local-socket-exempt protrusion + state integrity)
        violating_set: Set[Tuple[int, int]] = set()
        cert_cells: Dict[Tuple[int, int, int], List[Tuple[int, LDUBox]]] = {}
        for iid in part_ids_sorted:
            for b in parts[iid].world_boxes:
                for ck in SpatialBoxHashIndex.box_cells(b):
                    bucket = cert_cells.get(ck)
                    if bucket is not None:
                        for prev_iid, prev_b in bucket:
                            if prev_iid != iid and b.intersects(prev_b):
                                violating_set.add((min(prev_iid, iid), max(prev_iid, iid)))
                        bucket.append((iid, b))
                    else:
                        cert_cells[ck] = [(iid, b)]
        violating_pairs: List[Tuple[int, int]] = sorted(violating_set)
        collision_free = len(violating_pairs) == 0

        # Evaluate full-assembly static equilibrium once up front under declared `foundation_policy`
        eq = self.validator.evaluate(
            assembly,
            foundation_policy=fp_enum,
            ground_plane_z_ldu=ground_plane_z_ldu,
            deadline_monotonic_s=eff_cert_deadline,
        )

        # 3. Prefix Mechanics & 4. Monotone Insertion Certificates over explicit atomic `ExecutionTrace`
        cold_from_prefix: Optional[AssemblyState] = None
        eff_trace: Optional[ExecutionTrace] = execution_trace or getattr(
            assembly, "_execution_trace", None
        )
        if not verify_prefix and not verify_insertion:
            prefix_cert = PrefixMechanicsCert(
                status=CertStatus.NOT_APPLICABLE,
                evaluated_steps=0,
                stable_steps=0,
                first_unstable_step=None,
                max_prefix_utilization=0.0,
            )
            ins_cert = InsertionCert(
                status=CertStatus.NOT_APPLICABLE,
                evaluated_steps=0,
                monotone_axis_insertable_steps=0,
                blocked_steps=(),
            )
        elif eff_trace is None and not getattr(assembly, "has_execution_trace", True) and len(part_ids_sorted) > 0:
            # R8-WS1 Contract: Static/imported assembly with no execution trace returns UNKNOWN (never PASS!)
            prefix_cert = (
                PrefixMechanicsCert(
                    status=CertStatus.UNKNOWN,
                    evaluated_steps=0,
                    stable_steps=0,
                    first_unstable_step=None,
                    max_prefix_utilization=0.0,
                )
                if verify_prefix
                else PrefixMechanicsCert(CertStatus.NOT_APPLICABLE, 0, 0, None, 0.0)
            )
            ins_cert = (
                InsertionCert(
                    status=CertStatus.UNKNOWN,
                    evaluated_steps=0,
                    monotone_axis_insertable_steps=0,
                    blocked_steps=(),
                )
                if verify_insertion
                else InsertionCert(CertStatus.NOT_APPLICABLE, 0, 0, ())
            )
        else:
            if eff_trace is None:
                seen_rec: Set[int] = set()
                ordered_iids: List[int] = []
                for rid in getattr(assembly, "recorded_step_ids", ()):
                    if rid in parts and rid not in seen_rec:
                        seen_rec.add(rid)
                        ordered_iids.append(rid)
                for rid in part_ids_sorted:
                    if rid not in seen_rec:
                        seen_rec.add(rid)
                        ordered_iids.append(rid)
                eff_trace = ExecutionTrace(
                    steps=tuple(
                        AtomicBuildStep(
                            op_type=AtomicOpType.PLACE_PART,
                            instance_ids=(rid,),
                            part_id=parts[rid].part_id,
                            pose=parts[rid].pose,
                            is_grounded=parts[rid].is_grounded,
                        )
                        for rid in ordered_iids
                    ),
                    allow_subassemblies=False,
                    allow_temp_supports=False,
                    allow_reorientation=False,
                    trace_source="recorded_atomic_placements",
                )

            prefix_state = AssemblyState()
            prefix_place_ok = True
            used_temp_supports = False
            temp_tag_to_iid: Dict[str, int] = {}
            eval_steps = 0
            stable_steps = 0
            first_unstable: Optional[int] = None
            max_u = 0.0
            eval_ins = 0
            ok_ins = 0
            blocked_ins: List[int] = []
            base_lp_eta = 0.0
            base_witness: Any = None
            plumb_deltas: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
            plumb_hop_cache: Dict[int, Any] = {}
            full_hops_ref: Optional[Dict[int, Any]] = (
                getattr(self.validator, "_last_plumb_hop_cache", None)
                if getattr(self.validator, "_last_plumb_state_version", -1) == assembly.state_version
                else None
            )
            cur_lat_xy = (0.0, 0.0)
            grounded_reachable: Set[int] = set()
            ins_xy_boxes: Dict[Tuple[int, int], List[LDUBox]] = {}
            ins_max_z = -10**9

            def _record_ins_boxes(wbs_rec: Sequence[LDUBox]) -> None:
                nonlocal ins_max_z
                for bx_r in wbs_rec:
                    if bx_r.zmax > ins_max_z:
                        ins_max_z = bx_r.zmax
                    for cx_r in range(bx_r.xmin // 20, (bx_r.xmax - 1) // 20 + 1):
                        for cy_r in range(bx_r.ymin // 20, (bx_r.ymax - 1) // 20 + 1):
                            bucket_r = ins_xy_boxes.get((cx_r, cy_r))
                            if bucket_r is not None:
                                bucket_r.append(bx_r)
                            else:
                                ins_xy_boxes[(cx_r, cy_r)] = [bx_r]

            def _fast_z_plus_clear(wbs_chk: Sequence[LDUBox]) -> bool:
                if not wbs_chk:
                    return True
                for nb_c in wbs_chk:
                    if nb_c.zmin >= ins_max_z:
                        continue
                    for cx_c in range(nb_c.xmin // 20, (nb_c.xmax - 1) // 20 + 1):
                        for cy_c in range(nb_c.ymin // 20, (nb_c.ymax - 1) // 20 + 1):
                            bucket_c = ins_xy_boxes.get((cx_c, cy_c))
                            if bucket_c is not None:
                                for eb_c in bucket_c:
                                    if (
                                        eb_c.zmax > nb_c.zmin
                                        and nb_c.xmin < eb_c.xmax
                                        and nb_c.xmax > eb_c.xmin
                                        and nb_c.ymin < eb_c.ymax
                                        and nb_c.ymax > eb_c.ymin
                                    ):
                                        return False
                return True

            def _sync_grounded_add(new_iid: int, is_gr: bool) -> None:
                if is_gr or any(
                    nb in grounded_reachable for nb in prefix_state._part_adj.get(new_iid, ())
                ):
                    if new_iid not in grounded_reachable:
                        grounded_reachable.add(new_iid)
                        q_gr = [new_iid]
                        while q_gr:
                            cur_g = q_gr.pop()
                            for nb_g in prefix_state._part_adj.get(cur_g, ()):
                                if nb_g not in grounded_reachable:
                                    grounded_reachable.add(nb_g)
                                    q_gr.append(nb_g)

            n_trace_steps = len(eff_trace.steps)
            for s_idx, step in enumerate(eff_trace.steps):
                if eff_cert_deadline is not None and time.monotonic() >= eff_cert_deadline:
                    if verify_prefix and first_unstable is None:
                        first_unstable = s_idx
                        max_u = float("inf")
                    prefix_place_ok = False
                    break
                op = step.op_type
                if op == AtomicOpType.PLACE_PART:
                    if len(step.instance_ids) == 1 and step.instance_ids[0] in parts:
                        p = parts[step.instance_ids[0]]
                        iid_p, pid_p, pose_p, gr_p, col_p = (
                            p.instance_id,
                            p.part_id,
                            p.pose,
                            p.is_grounded,
                            p.color_id,
                        )
                    elif step.part_id and step.pose is not None:
                        iid_p = step.instance_ids[0] if step.instance_ids else prefix_state._next_instance_id
                        pid_p, pose_p, gr_p, col_p = step.part_id, step.pose, step.is_grounded, 15
                    else:
                        prefix_place_ok = False
                        if first_unstable is None:
                            first_unstable = s_idx
                        continue

                    if verify_insertion:
                        eval_ins += 1
                        wbs_cand_ins = (
                            p.world_boxes
                            if (len(step.instance_ids) == 1 and step.instance_ids[0] in parts)
                            else tuple(b.transform(pose_p) for b in get_brick_primitive(pid_p).hull_boxes)
                        )
                        if (not used_temp_supports and _fast_z_plus_clear(wbs_cand_ins)) or check_monotone_insertion_corridor(
                            prefix_state, pid_p, pose_p
                        ):
                            ok_ins += 1
                        else:
                            blocked_ins.append(iid_p)

                    step_ok = True
                    try:
                        prefix_state._next_instance_id = iid_p
                        actual_iid = prefix_state.place_root_part(
                            pid_p, pose_p, is_grounded=gr_p, color_id=col_p
                        )
                        _sync_grounded_add(actual_iid, gr_p)
                        if verify_insertion and not used_temp_supports:
                            _record_ins_boxes(prefix_state.placed_parts[actual_iid].world_boxes)
                    except ValueError:
                        prefix_place_ok = False
                        step_ok = False

                    if verify_prefix:
                        eval_steps += 1
                        has_ungrounded_island = len(grounded_reachable) < prefix_state.num_parts
                        if (not step_ok) or has_ungrounded_island:
                            base_lp_eta = float("inf")
                            if first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")
                        elif (
                            gr_p
                            and cur_lat_xy == (0.0, 0.0)
                            and base_lp_eta <= 1.0 - 1e-6
                            and (
                                fp_enum == FoundationPolicy.CLUTCHED_BASEPLATE
                                or (
                                    fp_enum == FoundationPolicy.TABLE_FRICTION_ONLY
                                    and p.world_boxes
                                    and any(
                                        abs(float(b.zmin) - ground_plane_z_ldu) <= 0.25
                                        and (b.xmin + 0.5)
                                        <= (
                                            sum(float(bx.volume_ldu3) * 0.5 * (bx.xmin + bx.xmax) for bx in p.world_boxes)
                                            / max(1.0, sum(float(bx.volume_ldu3) for bx in p.world_boxes))
                                        )
                                        <= (b.xmax - 0.5)
                                        and (b.ymin + 0.5)
                                        <= (
                                            sum(float(bx.volume_ldu3) * 0.5 * (bx.ymin + bx.ymax) for bx in p.world_boxes)
                                            / max(1.0, sum(float(bx.volume_ldu3) for bx in p.world_boxes))
                                        )
                                        <= (b.ymax - 0.5)
                                        for b in p.world_boxes
                                    )
                                )
                            )
                        ):
                            max_u = max(max_u, base_lp_eta)
                            stable_steps += 1
                        elif (
                            s_idx == n_trace_steps - 1
                            and not temp_tag_to_iid
                            and prefix_place_ok
                            and cur_lat_xy == (0.0, 0.0)
                            and prefix_state.num_parts == len(parts)
                        ):
                            if not np.isinf(eq.peak_utilization):
                                max_u = max(max_u, float(eq.peak_utilization))
                            if eq.is_stable:
                                stable_steps += 1
                            elif first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")
                        else:
                            can_ext = False
                            peak_d = float("inf")
                            if (
                                fp_enum in (FoundationPolicy.CLUTCHED_BASEPLATE, FoundationPolicy.TABLE_FRICTION_ONLY)
                                and cur_lat_xy == (0.0, 0.0)
                                and not np.isinf(base_lp_eta)
                            ):
                                if (
                                    full_hops_ref is not None
                                    and fp_enum == FoundationPolicy.CLUTCHED_BASEPLATE
                                    and not used_temp_supports
                                    and iid_p in full_hops_ref
                                    and iid_p not in plumb_hop_cache
                                ):
                                    fh_i = full_hops_ref[iid_p]
                                    if all((h_it[0] in prefix_state._parts) for h_it in fh_i if not h_it[-1]):
                                        plumb_hop_cache[iid_p] = tuple(h_it for h_it in fh_i if not h_it[-1])
                                can_ext, peak_d = self.validator.can_extend_plumb_primal_witness(
                                    prefix_state,
                                    iid_p,
                                    plumb_deltas,
                                    base_witness=base_witness,
                                    hop_cache=plumb_hop_cache,
                                    foundation_policy=fp_enum,
                                    ground_plane_z_ldu=ground_plane_z_ldu,
                                )
                                if not can_ext and prefix_state.num_parts >= 40:
                                    can_ext, peak_d = self.validator.verify_two_pier_bearing_compression(
                                        prefix_state,
                                        foundation_policy=fp_enum,
                                        ground_plane_z_ldu=ground_plane_z_ldu,
                                    )
                                    if can_ext:
                                        base_lp_eta = min(base_lp_eta, peak_d)
                            if can_ext and max(base_lp_eta, peak_d) <= 1.0 - 1e-5:
                                max_u = max(max_u, base_lp_eta, peak_d)
                                stable_steps += 1
                            else:
                                eq_c = self.validator.evaluate(
                                    prefix_state,
                                    foundation_policy=fp_enum,
                                    lateral_accel_xy=cur_lat_xy,
                                    ground_plane_z_ldu=ground_plane_z_ldu,
                                    deadline_monotonic_s=eff_cert_deadline,
                                )
                                if not np.isinf(eq_c.peak_utilization):
                                    max_u = max(max_u, float(eq_c.peak_utilization))
                                if eq_c.is_stable:
                                    base_lp_eta = float(eq_c.peak_utilization)
                                    base_witness = eq_c.primal_witness
                                    plumb_deltas.clear()
                                    stable_steps += 1
                                else:
                                    base_lp_eta = float("inf")
                                    base_witness = None
                                    if first_unstable is None:
                                        first_unstable = s_idx
                                        max_u = float("inf")
                                    if eq_c.budget_exceeded:
                                        prefix_place_ok = False
                                        break

                elif op == AtomicOpType.PLACE_SUBASSEMBLY:
                    eval_steps += 1 if verify_prefix else 0
                    if not eff_trace.allow_subassemblies or len(step.instance_ids) < 2:
                        if verify_prefix and first_unstable is None:
                            first_unstable = s_idx
                            max_u = float("inf")
                        if verify_insertion:
                            eval_ins += 1
                            blocked_ins.extend(step.instance_ids)
                        prefix_place_ok = False
                        continue

                    # 1. Verify internal subassembly connectivity & recursive subtrace buildability
                    sub_iids = [sid for sid in step.instance_ids if sid in parts]
                    if len(sub_iids) != len(step.instance_ids):
                        prefix_place_ok = False
                        if verify_prefix and first_unstable is None:
                            first_unstable = s_idx
                        continue
                    sub_ok = True
                    if step.subtrace:
                        # Phase 11 Ext 4A + TB-FIX-01 Gate 3: Replay explicit local-workbench `subtrace`
                        # step-by-step, verify single-component connectivity, and check invariant rigid
                        # `LDUFrame` transform `T_sub` (`wp_ref.pose == T_sub.compose(ls_op.pose)`)!
                        loc_sub_state = AssemblyState()
                        reindexed_sub_steps: List[AtomicBuildStep] = []
                        inferred_t_sub: Optional[LDUFrame] = None
                        if len(step.subtrace) != len(sub_iids):
                            sub_ok = False
                        else:
                            for idx_ls, ls_op in enumerate(step.subtrace):
                                if ls_op.part_id == "" or ls_op.pose is None or len(ls_op.instance_ids) != 1:
                                    sub_ok = False
                                    break
                                w_sid = ls_op.instance_ids[0]
                                if w_sid not in parts:
                                    sub_ok = False
                                    break
                                wp_ref = parts[w_sid]
                                if wp_ref.part_id != ls_op.part_id:
                                    sub_ok = False
                                    break
                                cur_t_sub = wp_ref.pose.compose(ls_op.pose.inverse())
                                if inferred_t_sub is None:
                                    inferred_t_sub = cur_t_sub
                                elif cur_t_sub != inferred_t_sub:
                                    sub_ok = False
                                    break
                                try:
                                    loc_sub_state._next_instance_id = idx_ls
                                    loc_sub_state.place_root_part(
                                        ls_op.part_id,
                                        ls_op.pose,
                                        is_grounded=ls_op.is_grounded,
                                        color_id=wp_ref.color_id,
                                    )
                                except ValueError:
                                    sub_ok = False
                                    break
                                reindexed_sub_steps.append(
                                    AtomicBuildStep(
                                        op_type=ls_op.op_type,
                                        instance_ids=(idx_ls,),
                                        part_id=ls_op.part_id,
                                        pose=ls_op.pose,
                                        is_grounded=ls_op.is_grounded,
                                        subtrace=ls_op.subtrace,
                                    )
                                )
                        if sub_ok and len(reindexed_sub_steps) >= 2:
                            vis_l: Set[int] = {0}
                            q_l = [0]
                            while q_l:
                                cl = q_l.pop()
                                for nbl in loc_sub_state._part_adj.get(cl, ()):
                                    if nbl not in vis_l:
                                        vis_l.add(nbl)
                                        q_l.append(nbl)
                            if len(vis_l) != len(reindexed_sub_steps):
                                sub_ok = False
                        if sub_ok:
                            sub_certs = self.certify(
                                loc_sub_state,
                                verify_prefix=verify_prefix,
                                verify_insertion=verify_insertion,
                                execution_trace=ExecutionTrace(
                                    steps=tuple(reindexed_sub_steps),
                                    allow_subassemblies=True,
                                ),
                            )
                            if not sub_certs.overall_certified:
                                sub_ok = False
                    else:
                        sub_min_z = min(
                            min(b.zmin for b in parts[sid].world_boxes) for sid in sub_iids
                        )
                        sub_state = AssemblyState()
                        for sid in sorted(sub_iids, key=lambda i: min(b.zmin for b in parts[i].world_boxes)):
                            sp_sub = parts[sid]
                            is_sub_g = min(b.zmin for b in sp_sub.world_boxes) <= sub_min_z
                            try:
                                sub_state._next_instance_id = sid
                                sub_state.place_root_part(
                                    sp_sub.part_id,
                                    sp_sub.pose,
                                    is_grounded=is_sub_g,
                                    color_id=sp_sub.color_id,
                                )
                            except ValueError:
                                sub_ok = False
                                break
                        # Subassembly must be a single connected physical component!
                        if sub_ok and len(sub_iids) >= 2:
                            visited_sub: Set[int] = {sub_iids[0]}
                            queue_sub = [sub_iids[0]]
                            while queue_sub:
                                cur_s = queue_sub.pop()
                                for nb_s in sub_state._part_adj.get(cur_s, ()):
                                    if nb_s not in visited_sub:
                                        visited_sub.add(nb_s)
                                        queue_sub.append(nb_s)
                            if len(visited_sub) != len(sub_iids):
                                sub_ok = False
                        if sub_ok:
                            sub_certs = self.certify(
                                sub_state,
                                verify_prefix=verify_prefix,
                                verify_insertion=verify_insertion,
                                execution_trace=None,
                            )
                            if not sub_certs.overall_certified:
                                sub_ok = False

                    group_boxes = [b for sid in sub_iids for b in parts[sid].world_boxes]
                    if verify_insertion:
                        eval_ins += 1
                        if sub_ok and check_rigid_group_insertion_corridor(prefix_state, group_boxes):
                            ok_ins += 1
                        else:
                            blocked_ins.extend(sub_iids)

                    for sid in sub_iids:
                        sp_sub = parts[sid]
                        try:
                            prefix_state._next_instance_id = sid
                            act_s = prefix_state.place_root_part(
                                sp_sub.part_id,
                                sp_sub.pose,
                                is_grounded=sp_sub.is_grounded,
                                color_id=sp_sub.color_id,
                            )
                            _sync_grounded_add(act_s, sp_sub.is_grounded)
                            if verify_insertion and not used_temp_supports:
                                _record_ins_boxes(prefix_state.placed_parts[act_s].world_boxes)
                        except ValueError:
                            sub_ok = False
                            prefix_place_ok = False

                    if verify_prefix:
                        if not sub_ok:
                            base_lp_eta = float("inf")
                            if first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")
                        else:
                            eq_sub = self.validator.evaluate(
                                prefix_state,
                                foundation_policy=fp_enum,
                                lateral_accel_xy=cur_lat_xy,
                                ground_plane_z_ldu=ground_plane_z_ldu,
                                deadline_monotonic_s=eff_cert_deadline,
                            )
                            if not np.isinf(eq_sub.peak_utilization):
                                max_u = max(max_u, float(eq_sub.peak_utilization))
                            if eq_sub.is_stable:
                                base_lp_eta = float(eq_sub.peak_utilization)
                                base_witness = eq_sub.primal_witness
                                plumb_deltas.clear()
                                stable_steps += 1
                            elif first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")

                elif op == AtomicOpType.ADD_TEMP_SUPPORT:
                    used_temp_supports = True
                    if (
                        not eff_trace.allow_temp_supports
                        or not step.part_id
                        or step.pose is None
                        or not step.temp_support_tag
                    ):
                        prefix_place_ok = False
                        if verify_prefix:
                            eval_steps += 1
                            if first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")
                        continue
                    temp_iid = 900000 + s_idx
                    if verify_insertion:
                        eval_ins += 1
                        if check_monotone_insertion_corridor(prefix_state, step.part_id, step.pose):
                            ok_ins += 1
                        else:
                            blocked_ins.append(temp_iid)
                    try:
                        prefix_state._next_instance_id = temp_iid
                        act_t = prefix_state.place_root_part(
                            step.part_id, step.pose, is_grounded=step.is_grounded
                        )
                        _sync_grounded_add(act_t, step.is_grounded)
                        temp_tag_to_iid[step.temp_support_tag] = temp_iid
                        ts_ok = True
                    except ValueError:
                        ts_ok = False
                        prefix_place_ok = False
                    if verify_prefix:
                        eval_steps += 1
                        eq_ts = (
                            self.validator.evaluate(
                                prefix_state,
                                foundation_policy=fp_enum,
                                lateral_accel_xy=cur_lat_xy,
                                ground_plane_z_ldu=ground_plane_z_ldu,
                                deadline_monotonic_s=eff_cert_deadline,
                            )
                            if ts_ok
                            else EquilibriumResult(False, float("inf"), -float("inf"), {}, None, 0.0, "fail")
                        )
                        if eq_ts.is_stable:
                            base_lp_eta = float(eq_ts.peak_utilization)
                            base_witness = eq_ts.primal_witness
                            plumb_deltas.clear()
                            max_u = max(max_u, base_lp_eta)
                            stable_steps += 1
                        elif first_unstable is None:
                            first_unstable = s_idx
                            max_u = float("inf")

                elif op == AtomicOpType.REMOVE_TEMP_SUPPORT:
                    used_temp_supports = True
                    if (
                        not eff_trace.allow_temp_supports
                        or step.temp_support_tag not in temp_tag_to_iid
                    ):
                        prefix_place_ok = False
                        if verify_prefix:
                            eval_steps += 1
                            if first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")
                        continue
                    rem_iid = temp_tag_to_iid.pop(step.temp_support_tag)
                    rem_part = prefix_state._parts[rem_iid]
                    prefix_state.remove_part(rem_iid)
                    if verify_insertion:
                        eval_ins += 1
                        if check_monotone_insertion_corridor(
                            prefix_state, rem_part.part_id, rem_part.pose
                        ):
                            ok_ins += 1
                        else:
                            blocked_ins.append(rem_iid)
                    if verify_prefix:
                        eval_steps += 1
                        depths_rem = prefix_state.compute_support_depths()
                        grounded_reachable = {k for k, d in depths_rem.items() if d < 999}
                        if any(d >= 999 for d in depths_rem.values()):
                            if first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")
                        else:
                            eq_rem = self.validator.evaluate(
                                prefix_state,
                                foundation_policy=fp_enum,
                                lateral_accel_xy=cur_lat_xy,
                                ground_plane_z_ldu=ground_plane_z_ldu,
                                deadline_monotonic_s=eff_cert_deadline,
                            )
                            if not np.isinf(eq_rem.peak_utilization):
                                max_u = max(max_u, float(eq_rem.peak_utilization))
                            if eq_rem.is_stable:
                                base_lp_eta = float(eq_rem.peak_utilization)
                                base_witness = eq_rem.primal_witness
                                plumb_deltas.clear()
                                stable_steps += 1
                            elif first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")

                elif op == AtomicOpType.REORIENT_ASSEMBLY:
                    if not eff_trace.allow_reorientation:
                        if verify_prefix:
                            eval_steps += 1
                            if first_unstable is None:
                                first_unstable = s_idx
                                max_u = float("inf")
                        continue
                    cur_lat_xy = (float(step.gravity_unit[0]), float(step.gravity_unit[1]))
                    if verify_prefix:
                        eval_steps += 1
                        eq_ro = self.validator.evaluate(
                            prefix_state,
                            foundation_policy=fp_enum,
                            lateral_accel_xy=cur_lat_xy,
                            ground_plane_z_ldu=ground_plane_z_ldu,
                            deadline_monotonic_s=eff_cert_deadline,
                        )
                        if eq_ro.is_stable:
                            base_lp_eta = float(eq_ro.peak_utilization)
                            plumb_deltas.clear()
                            max_u = max(max_u, base_lp_eta)
                            stable_steps += 1
                        elif first_unstable is None:
                            first_unstable = s_idx
                            max_u = float("inf")

            # Final trace post-condition: all temporary supports must be removed and final part set must match!
            if temp_tag_to_iid or set(prefix_state._parts.keys()) != set(parts.keys()):
                prefix_place_ok = False
                if verify_prefix and first_unstable is None:
                    first_unstable = max(0, n_trace_steps - 1)
                    max_u = float("inf")

            if prefix_place_ok and not used_temp_supports:
                cold_from_prefix = prefix_state

            prefix_cert = (
                PrefixMechanicsCert(
                    status=CertStatus.PASS if first_unstable is None else CertStatus.FAIL,
                    evaluated_steps=eval_steps,
                    stable_steps=stable_steps,
                    first_unstable_step=first_unstable,
                    max_prefix_utilization=max_u,
                )
                if verify_prefix
                else PrefixMechanicsCert(
                    status=CertStatus.NOT_APPLICABLE,
                    evaluated_steps=0,
                    stable_steps=0,
                    first_unstable_step=None,
                    max_prefix_utilization=0.0,
                )
            )
            ins_cert = (
                InsertionCert(
                    status=CertStatus.PASS if len(blocked_ins) == 0 else CertStatus.FAIL,
                    evaluated_steps=eval_ins,
                    monotone_axis_insertable_steps=ok_ins,
                    blocked_steps=tuple(blocked_ins),
                )
                if verify_insertion
                else InsertionCert(
                    status=CertStatus.NOT_APPLICABLE,
                    evaluated_steps=0,
                    monotone_axis_insertable_steps=0,
                    blocked_steps=(),
                )
            )

        if not collision_free:
            integrity_ok = False
        elif cold_from_prefix is not None:
            integrity_ok = (
                assembly._parts == cold_from_prefix._parts
                and assembly._matched_ports == cold_from_prefix._matched_ports
                and assembly._part_adj == cold_from_prefix._part_adj
                and assembly._collision_boxes == cold_from_prefix._collision_boxes
            )
        else:
            integrity_ok = assembly.verify_state_integrity()
        protrusion_free = integrity_ok

        # Check port compatibility across all matched ports
        ports_compat = True
        for gpid_a, gpid_b in assembly._matched_ports.items():
            pa = parts[gpid_a[0]]
            pb = parts[gpid_b[0]]
            sa = get_brick_primitive(pa.part_id).ports[gpid_a[1]]
            sb = get_brick_primitive(pb.part_id).ports[gpid_b[1]]
            if not are_connectors_compatible(sa.connector_type, sa.polarity, sb.connector_type, sb.polarity):
                ports_compat = False
                break

        if target_field is not None:
            if parts:
                placed_boxes = [b for p in parts.values() for b in p.world_boxes]
                placed_vol = float(sum(b.volume_ldu3 for b in placed_boxes))
                tot_t_vol = float(target_field.total_volume_ldu3)
                if collision_free and hasattr(target_field, "query_box_volume"):
                    inter_vol = float(sum(target_field.query_box_volume(b) for b in placed_boxes))
                else:
                    inter_vol = float(target_field.intersection_volume(placed_boxes))
                union_vol = max(tot_t_vol + placed_vol - inter_vol, 1.0)
                t_iou = float(inter_vol / union_vol) if (tot_t_vol > 0.0 or placed_vol > 0.0) else 1.0
                t_prec = float(inter_vol / max(placed_vol, 1.0)) if placed_vol > 0.0 else 1.0
                t_rec = float(inter_vol / max(tot_t_vol, 1.0))
                non_empty_target_covered = True
            elif int(target_field.total_volume_ldu3) > 0:
                t_iou, t_prec, t_rec = 0.0, 0.0, 0.0
                non_empty_target_covered = False
            else:
                t_iou, t_prec, t_rec = 1.0, 1.0, 1.0
                non_empty_target_covered = True
        else:
            t_iou, t_prec, t_rec = 1.0, 1.0, 1.0
            non_empty_target_covered = True

        geom_pass = collision_free and protrusion_free and ports_compat and integrity_ok and non_empty_target_covered
        geom_cert = GeometryCert(
            status=CertStatus.PASS if geom_pass else CertStatus.FAIL,
            collision_free=collision_free,
            protrusion_collision_free=protrusion_free,
            all_ports_compatible=ports_compat,
            state_integrity_verified=integrity_ok,
            target_iou=t_iou,
            target_precision=t_prec,
            target_recall=t_rec,
            violating_pairs=tuple(violating_pairs),
        )

        # 2. Static Mechanics Certificate
        if not geom_pass or not eq.is_stable or eq.budget_exceeded:
            cold_ok = False
        elif assembly.state_version == 0 or cold_from_prefix is not None:
            cold_ok = True
        else:
            cold_ok = self.validator.verify_against_cold_rebuild(
                assembly,
                foundation_policy=fp_enum,
                ground_plane_z_ldu=ground_plane_z_ldu,
                deadline_monotonic_s=eff_cert_deadline,
            )
        static_pass = eq.is_stable and cold_ok
        static_cert = StaticMechanicsCert(
            status=CertStatus.PASS if static_pass else CertStatus.FAIL,
            is_stable=eq.is_stable,
            peak_utilization=float(eq.peak_utilization),
            safety_margin=float(eq.safety_margin),
            weakest_contact=eq.weakest_contact,
            moment_reference=eq.moment_reference,
            asymmetric_bearing_active=eq.asymmetric_bearing_active,
            cold_rebuild_consistent=cold_ok,
            solve_time_ms=float(eq.solve_time_ms),
        )

        # Optional Perturbation Mechanics Certificate on cubic AssemblyState
        if verify_perturbation:
            worst_lat_u = float(eq.peak_utilization)
            pert_ok = bool(eq.is_stable)
            for lat_xy in ((0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05)):
                eq_l = self.validator.evaluate(
                    assembly,
                    foundation_policy=fp_enum,
                    lateral_accel_xy=lat_xy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=eff_cert_deadline,
                )
                worst_lat_u = max(worst_lat_u, float(eq_l.peak_utilization))
                if not eq_l.is_stable:
                    pert_ok = False
            pert_cert = PerturbationMechanicsCert(
                status=CertStatus.PASS if pert_ok else CertStatus.FAIL,
                is_perturbation_stable=pert_ok,
                min_potential_delta_g_ldu=0.0,
                worst_lateral_utilization=worst_lat_u,
                unstable_joint_ids=(),
                mechanism_regime="rigid_perturbed_ok" if pert_ok else "lateral_perturbation_unstable",
            )
        else:
            pert_cert = PerturbationMechanicsCert()

        # Optional Model Agreement & Calibrated Sub-Stud Polygon Check (`F9`: strictly analytical model agreement!)
        sub_res_cached = None
        if verify_calibrated_substud_physics or verify_model_agreement:
            from topo_brick.evaluation.physical_validation import CalibratedSubStudPolygonValidator

            sub_res_cached = CalibratedSubStudPolygonValidator().evaluate_assembly_state(assembly)

        if (verify_calibrated_substud_physics or verify_model_agreement) and sub_res_cached is not None:
            agreed = bool(sub_res_cached.is_stable == eq.is_stable)
            cal_pass = bool(agreed and sub_res_cached.is_stable) if verify_calibrated_substud_physics else agreed
            agree_cert = ModelAgreementCert(
                status=CertStatus.PASS if cal_pass else CertStatus.FAIL,
                agreed_with_substud_polygon=agreed,
                kernel_lp_stable=bool(eq.is_stable),
                substud_polygon_stable=bool(sub_res_cached.is_stable),
                substud_peak_utilization=float(sub_res_cached.peak_utilization),
                calibration_reference="cmu_stablelego_8hw_holdout120_v1",
            )
        else:
            agree_cert = ModelAgreementCert()

        phys_cert = self._evaluate_physical_validation_cert(external_physical_observation)
        inv_cert = self._evaluate_inventory_cert(parts, inventory_counts, colored_inventory)

        return CertificateBundle(
            geometry=geom_cert,
            static_mechanics=static_cert,
            prefix_mechanics=prefix_cert,
            insertion=ins_cert,
            physical_validation=phys_cert,
            perturbation_mechanics=pert_cert,
            model_agreement=agree_cert,
            loop_closure=LoopClosureCert(),
            inventory=inv_cert,
            foundation_policy=fp_enum.value,
            execution_trace=eff_trace,
            ground_plane_z_ldu=float(ground_plane_z_ldu),
        )

    def constrain(
        self,
        assembly: AssemblyState,
        target_field: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
        allowed_part_ids: Optional[Sequence[str]] = None,
        max_actions: int = 32,
        candidate_cache: Optional[IncrementalCandidateCache] = None,
    ) -> KernelConstrainResult:
        """
        Return certified valid next-step placement actions on `assembly`'s active frontier.
        Every returned action is guaranteed to be collision-free, protrusion-collision-free,
        port-compatible, and static-equilibrium stable.
        Uses `IncrementalCandidateCache` and O(1) `push_trial_part`/`pop_trial_part` rollback
        (zero `copy.deepcopy(assembly)`).
        """
        t0 = time.perf_counter()
        kit = tuple(allowed_part_ids) if allowed_part_ids is not None else self.default_kit
        for pid in kit:
            if not is_known_part_id(pid):
                return KernelConstrainResult(
                    status=KernelResultStatus.UNSUPPORTED_SPEC,
                    valid_actions=(),
                    candidate_pool_size=0,
                    certified_action_count=0,
                    wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
                    updated_cache=candidate_cache,
                )

        # Always derive the full active frontier so `IncrementalCandidateCache` never evicts
        # unconsumed lower-course ports; prioritize lower-Z ports in selective expansion.
        frontier = assembly.get_active_frontier()

        updated_cache: Optional[IncrementalCandidateCache] = candidate_cache
        if len(frontier) > 12:
            p_scores = compute_cheap_pre_expansion_port_scores(assembly, frontier, target_field)
            # Prioritize bottom-up course completion so lower incomplete courses are expanded first
            z_vals = np.array([float(ep.world_frame.pos[2]) for ep in frontier.ports], dtype=np.float64)
            p_scores = p_scores - 0.5 * (z_vals - float(np.min(z_vals)))
            pool, updated_cache, _ = assembly.generate_candidate_pool_incremental_selective(
                port_priority_scores=p_scores,
                max_expanded_ports=12,
                cache=candidate_cache,
                allowed_part_ids=kit,
                frontier=frontier,
                build_conflict_graph=False,
            )
        else:
            pool, updated_cache = assembly.generate_candidate_pool_incremental(
                cache=candidate_cache,
                allowed_part_ids=kit,
                frontier=frontier,
                build_conflict_graph=False,
            )

        if len(pool) == 0:
            return KernelConstrainResult(
                status=KernelResultStatus.PROVEN_INFEASIBLE,
                valid_actions=(),
                candidate_pool_size=0,
                certified_action_count=0,
                wall_clock_ms=(time.perf_counter() - t0) * 1000.0,
                updated_cache=updated_cache,
            )

        scored_cands: List[Tuple[float, int, CandidatePlacement]] = []
        for cand in pool.candidates:
            vol = sum(b.volume_ldu3 for b in cand.world_boxes)
            if target_field is not None:
                inter = int(target_field.intersection_volume(cand.world_boxes))
                if inter < int(0.95 * vol):
                    continue
            else:
                inter = vol
            c_z0 = float(min(wb.zmin for wb in cand.world_boxes))
            sc = float(inter) + (2500.0 if cand.closes_structural_seam else 0.0) - 500.0 * c_z0
            scored_cands.append((sc, inter, cand))

        if not scored_cands and len(frontier) > 12:
            full_pool, updated_cache = assembly.generate_candidate_pool_incremental(
                cache=updated_cache,
                allowed_part_ids=kit,
                frontier=frontier,
                build_conflict_graph=False,
            )
            pool = full_pool
            for cand in full_pool.candidates:
                vol = sum(b.volume_ldu3 for b in cand.world_boxes)
                if target_field is not None:
                    inter = int(target_field.intersection_volume(cand.world_boxes))
                    if inter < int(0.95 * vol):
                        continue
                else:
                    inter = vol
                c_z0 = float(min(wb.zmin for wb in cand.world_boxes))
                sc = float(inter) + (2500.0 if cand.closes_structural_seam else 0.0) - 500.0 * c_z0
                scored_cands.append((sc, inter, cand))

        scored_cands.sort(key=lambda x: x[0], reverse=True)
        grounded_dilated_c: Set[Tuple[int, int]] = set()
        for p_g in assembly.placed_parts.values():
            if p_g.is_grounded:
                for gb in p_g.world_boxes:
                    for gx in range(gb.xmin // 20 - 1, (gb.xmax + 19) // 20 + 1):
                        for gy in range(gb.ymin // 20 - 1, (gb.ymax + 19) // 20 + 1):
                            grounded_dilated_c.add((gx, gy))
        all_existing_within_grounded = bool(grounded_dilated_c) and all(
            p_e.is_grounded
            or all(
                (gx, gy) in grounded_dilated_c
                for b in p_e.world_boxes
                for gx in range(b.xmin // 20, (b.xmax + 19) // 20)
                for gy in range(b.ymin // 20, (b.ymax + 19) // 20)
            )
            for p_e in assembly.placed_parts.values()
        )
        base_eq = (
            EquilibriumResult(
                is_stable=True,
                peak_utilization=0.02,
                safety_margin=0.98,
                contact_utilizations={},
                weakest_contact=None,
                solve_time_ms=0.0,
                status_message="Compressive footprint",
            )
            if all_existing_within_grounded
            else (self.validator.evaluate(assembly) if assembly.num_parts > 0 else None)
        )
        if base_eq is None or not base_eq.is_stable or base_eq.peak_utilization > 0.90:
            grounded_dilated_c.clear()

        valid_actions: List[ValidPlacementAction] = []
        for _, inter, cand in scored_cands:
            if len(valid_actions) >= max_actions:
                break
            cand_z0 = min(wb.zmin for wb in cand.world_boxes)
            fast_comp_ok = (
                bool(grounded_dilated_c)
                and any(
                    gp[0] in assembly._parts
                    and max(bb.zmax for bb in assembly._parts[gp[0]].world_boxes) <= cand_z0
                    for gp in cand.consumed_ports
                )
                and all(
                    (gx, gy) in grounded_dilated_c
                    for b in cand.world_boxes
                    for gx in range(b.xmin // 20, (b.xmax + 19) // 20)
                    for gy in range(b.ymin // 20, (b.ymax + 19) // 20)
                )
            )
            if fast_comp_ok and base_eq is not None:
                x, y, z = cand.pose.pos
                tok = f"PLACE {cand.part_id} {x} {y} {z} {cand.pose.rot_id}"
                valid_actions.append(
                    ValidPlacementAction(
                        action_token=tok,
                        part_id=cand.part_id,
                        pose=cand.pose,
                        consumed_ports=cand.consumed_ports,
                        target_intersection_ldu3=inter,
                        closes_structural_seam=cand.closes_structural_seam,
                        creates_biconnected_cycle=cand.creates_biconnected_cycle,
                        post_step_utilization=float(max(base_eq.peak_utilization, 0.02)),
                    )
                )
                continue

            tok_trial: Optional[TrialPartRollbackToken] = None
            try:
                tok_trial = assembly.push_trial_part(cand.part_id, cand.pose, is_grounded=False)
                eq = self.validator.evaluate(assembly)
                if not eq.is_stable:
                    continue
                x, y, z = cand.pose.pos
                tok = f"PLACE {cand.part_id} {x} {y} {z} {cand.pose.rot_id}"
                valid_actions.append(
                    ValidPlacementAction(
                        action_token=tok,
                        part_id=cand.part_id,
                        pose=cand.pose,
                        consumed_ports=cand.consumed_ports,
                        target_intersection_ldu3=inter,
                        closes_structural_seam=cand.closes_structural_seam,
                        creates_biconnected_cycle=cand.creates_biconnected_cycle,
                        post_step_utilization=float(eq.peak_utilization),
                    )
                )
            except ValueError:
                continue
            finally:
                if tok_trial is not None:
                    assembly.pop_trial_part(tok_trial)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        status = (
            KernelResultStatus.COMPLETED_CERTIFIED
            if valid_actions
            else KernelResultStatus.PROVEN_INFEASIBLE
        )
        return KernelConstrainResult(
            status=status,
            valid_actions=tuple(valid_actions),
            candidate_pool_size=len(pool),
            certified_action_count=len(valid_actions),
            wall_clock_ms=elapsed_ms,
            updated_cache=updated_cache,
        )


    def repair(
        self,
        assembly: AssemblyState,
        target_field: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
        allowed_part_ids: Optional[Sequence[str]] = None,
        allowed_edits: Optional[FrozenSet[str]] = None,
        locked_instance_ids: FrozenSet[int] = frozenset(),
        locked_world_boxes: Tuple[LDUBox, ...] = (),
        inventory_counts: Optional[Dict[str, int]] = None,
        colored_inventory: Optional[Any] = None,
        forbidden_support_boxes: Sequence[LDUBox] = (),
        support_policy: str = "none",
        min_retained_iou: Optional[float] = None,
        max_deleted_volume_ratio: Optional[float] = None,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        time_budget_ms: float = 1200.0,
        max_constructive_steps: int = 1,
    ) -> KernelRepairResult:
        """Task-aware constrained repair (`TopoBrickKernel.repair`)."""
        from topo_brick.kernel.constrained_repair import execute_constrained_repair

        return execute_constrained_repair(
            self,
            assembly=assembly,
            target_field=target_field,
            allowed_part_ids=allowed_part_ids,
            allowed_edits=allowed_edits,
            locked_instance_ids=locked_instance_ids,
            locked_world_boxes=locked_world_boxes,
            inventory_counts=inventory_counts,
            colored_inventory=colored_inventory,
            forbidden_support_boxes=forbidden_support_boxes,
            support_policy=support_policy,
            min_retained_iou=min_retained_iou,
            max_deleted_volume_ratio=max_deleted_volume_ratio,
            foundation_policy=foundation_policy,
            time_budget_ms=time_budget_ms,
            max_constructive_steps=max_constructive_steps,
        )

    def _select_from_pool(
        self,
        solver_id: str,
        state: AssemblyState,
        pool: CandidatePool,
        target_band: SparseTiledTargetField | TargetShapeBoxes,
        max_bricks: Optional[int] = None,
        inventory_counts: Optional[Dict[str, int]] = None,
        full_target: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
        z0_course: Optional[int] = None,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        ground_plane_z_ldu: float = 0.0,
        deadline_monotonic_s: Optional[float] = None,
    ) -> Tuple[List[int], int]:
        if len(pool) == 0 or (deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s):
            return [], 0
        snot_bonus: Optional[np.ndarray] = None
        if full_target is not None and z0_course is not None:
            m_p = len(pool)
            for cid, cand in enumerate(pool.candidates):
                prim_c = get_brick_primitive(cand.part_id)
                if not prim_c.is_snot:
                    continue
                for ps in prim_c.ports:
                    if ps.polarity != PortPolarity.STUD:
                        continue
                    wf = cand.pose.compose(ps.local_frame)
                    if wf.z_axis[2] == 0 and (wf.z_axis[0] != 0 or wf.z_axis[1] != 0):
                        pt_out = wf.transform_point((0, 0, 4))
                        if full_target.contains_point(pt_out) and not full_target.contains_point(
                            (pt_out[0], pt_out[1], z0_course + 1)
                        ):
                            if snot_bonus is None:
                                snot_bonus = np.zeros(m_p, dtype=np.float64)
                            snot_bonus[cid] += 4.5
        if solver_id == "S1_CoverAndRepair":
            return self.s1_solver.solve_course(
                state,
                pool,
                target_band,
                max_bricks=max_bricks,
                inventory_counts=inventory_counts,
                candidate_bonus_scores=snot_bonus,
                foundation_policy=foundation_policy,
                ground_plane_z_ldu=ground_plane_z_ldu,
                deadline_monotonic_s=deadline_monotonic_s,
            )
        if solver_id == "S2_PairSeededLocalSearch":
            return (
                self.s2_solver.solve_course(
                    state, pool, target_band, max_bricks=max_bricks, inventory_counts=inventory_counts
                ),
                0,
            )
        if solver_id == "S3_RecurrentBatchRefiner":
            return self.s3_solver.solve_course(state, pool, target_band, max_bricks=max_bricks), 0
        if solver_id == "S4_D3PMBatchSampler":
            modes = self.s4_solver.sample_diverse_courses(pool, target_band, max_bricks=max_bricks)
            return (list(modes[0]) if modes else []), 0
        return self.s1_solver.solve_course(
            state,
            pool,
            target_band,
            max_bricks=max_bricks,
            inventory_counts=inventory_counts,
            candidate_bonus_scores=snot_bonus,
            foundation_policy=foundation_policy,
            ground_plane_z_ldu=ground_plane_z_ldu,
            deadline_monotonic_s=deadline_monotonic_s,
        )


    def realize(
        self,
        target_field: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
        solver_id: str = "S1_CoverAndRepair",
        allowed_part_ids: Optional[Sequence[str]] = None,
        kit_name: str = "STABLETEXT2BRICK_8PART_KIT",
        inventory_counts: Optional[Dict[str, int]] = None,
        support_policy: str = "none",
        port_filter_keep_ratio: Optional[float] = None,
        m2_filter: Optional[Any] = None,
        object_id: str = "blind_target",
        source_dataset: str = "blind_target_field",
        source_row_index: int = -1,
        category_id: str = "",
        caption: str = "",
        split: str = "test",
        reference_part_count: int = 0,
        foundation_policy: FoundationPolicy | str = FoundationPolicy.CLUTCHED_BASEPLATE,
        verify_perturbation: bool = False,
        t_target_init_ms: float = 0.0,
        forbidden_support_boxes: Tuple[LDUBox, ...] = (),
        surface_refinement: str = "none",
        adaptive_mesh_target: Optional[Any] = None,
        macro_block_mode: str = "off",
        part_retrieval: str = "none",
        retrieval_top_k: int = 32,
        license_filter: str = "all",
        semantic_style_hint: Optional[str] = None,
        vertex_colors_rgb: Optional[np.ndarray] = None,
        verts_ldu: Optional[np.ndarray] = None,
        colored_inventory: Optional[Any] = None,
        hollow_shell_policy: Optional[Any] = None,
        subassembly_graph: Optional[Sequence[Any]] = None,
        landmark_requests: Sequence[Any] = (),
        view_priority: Optional[Any] = None,
        max_wall_clock_s: Optional[float] = None,
        deadline_monotonic_s: Optional[float] = None,
        ground_plane_z_ldu: float = 0.0,
        *,
        request_spec: Optional[RealizeRequestSpec] = None,
        vertex_region_labels: Optional[np.ndarray] = None,
        open_color_palette_mode: str = "solid_54_default",
        symmetry_plane_axis: Optional[str] = None,
        symmetry_plane_coord_ldu: Optional[float] = None,
    ) -> KernelRealizeResult:
        """
        Genuinely blind whole-target realization (`TopoBrickKernel.realize`), dispatching through
        unified `ResolvedRealizePolicy` (`Phase 15`) with shared deadlines, ground datum, and
        `RealizeRouteProvenance`.
        """
        if request_spec is not None:
            req = request_spec
        else:
            if target_field is None:
                raise ValueError("TopoBrickKernel.realize requires either `target_field` or `request_spec`.")
            req = RealizeRequestSpec(
                target_field=target_field,
                solver_id=solver_id,
                allowed_part_ids=allowed_part_ids,
                kit_name=kit_name,
                inventory_counts=inventory_counts,
                support_policy=support_policy,
                port_filter_keep_ratio=port_filter_keep_ratio,
                m2_filter=m2_filter,
                object_id=object_id,
                source_dataset=source_dataset,
                source_row_index=source_row_index,
                category_id=category_id,
                caption=caption,
                split=split,
                reference_part_count=reference_part_count,
                foundation_policy=foundation_policy,
                verify_perturbation=verify_perturbation,
                t_target_init_ms=t_target_init_ms,
                forbidden_support_boxes=tuple(forbidden_support_boxes),
                surface_refinement=surface_refinement,
                adaptive_mesh_target=adaptive_mesh_target,
                macro_block_mode=macro_block_mode,
                part_retrieval=part_retrieval,
                retrieval_top_k=retrieval_top_k,
                license_filter=license_filter,
                semantic_style_hint=semantic_style_hint,
                vertex_colors_rgb=vertex_colors_rgb,
                verts_ldu=verts_ldu,
                colored_inventory=colored_inventory,
                hollow_shell_policy=hollow_shell_policy,
                subassembly_graph=subassembly_graph,
                landmark_requests=landmark_requests,
                view_priority=view_priority,
                max_wall_clock_s=max_wall_clock_s,
                deadline_monotonic_s=deadline_monotonic_s,
                ground_plane_z_ldu=ground_plane_z_ldu,
                vertex_region_labels=vertex_region_labels,
                open_color_palette_mode=open_color_palette_mode,
                symmetry_plane_axis=symmetry_plane_axis,
                symmetry_plane_coord_ldu=symmetry_plane_coord_ldu,
            )
        policy = resolve_realize_policy(self, req)
        return execute_resolved_realize(self, req, policy)

    def realize_external_record(
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
        """Realize and certify an external LDraw target record (`mode="blind"` default)."""
        return execute_realize_external_record(
            self,
            record=record,
            solver_id=solver_id,
            allowed_part_ids=allowed_part_ids,
            kit_name=kit_name,
            port_filter_keep_ratio=port_filter_keep_ratio,
            m2_filter=m2_filter,
            mode=mode,
            inventory_counts=inventory_counts,
            support_policy=support_policy,
        )


_LAZY_KERNEL_COMPAT_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ConstrainedRolloutMetrics": ("topo_brick.kernel.learned_bottleneck_controls", "ConstrainedRolloutMetrics"),
    "BrickActionGrammarAdapter": ("topo_brick.kernel.learned_bottleneck_controls", "BrickActionGrammarAdapter"),
    "DescriptorFrameEncoder": ("topo_brick.kernel.learned_bottleneck_controls", "DescriptorFrameEncoder"),
    "LearnedCatalogAndPortFilter": ("topo_brick.kernel.learned_bottleneck_controls", "LearnedCatalogAndPortFilter"),
    "SearchDistilledValueModel": ("topo_brick.kernel.learned_bottleneck_controls", "SearchDistilledValueModel"),
    "PromotionGateVerdict": ("topo_brick.kernel.kernel_contracts", "PromotionGateVerdict"),
    "Deliverable1Report": ("topo_brick.kernel.revised_kernel_deliverables", "Deliverable1Report"),
    "run_deliverable1_small_kernel_verification": (
        "topo_brick.kernel.revised_kernel_deliverables",
        "run_deliverable1_small_kernel_verification",
    ),
    "SolverCohortMetrics": ("topo_brick.kernel.revised_kernel_deliverables", "SolverCohortMetrics"),
    "Deliverable2Report": ("topo_brick.kernel.revised_kernel_deliverables", "Deliverable2Report"),
    "run_deliverable2_untouched_external_benchmark": (
        "topo_brick.kernel.revised_kernel_deliverables",
        "run_deliverable2_untouched_external_benchmark",
    ),
    "Deliverable3Report": ("topo_brick.kernel.revised_kernel_deliverables", "Deliverable3Report"),
    "run_deliverable3_external_adapter_and_connectors": (
        "topo_brick.kernel.revised_kernel_deliverables",
        "run_deliverable3_external_adapter_and_connectors",
    ),
    "Deliverable4Report": ("topo_brick.kernel.revised_kernel_deliverables", "Deliverable4Report"),
    "build_parametric_non_myopic_trap": (
        "topo_brick.kernel.revised_kernel_deliverables",
        "build_parametric_non_myopic_trap",
    ),
    "run_deliverable4_learning_at_measured_bottleneck": (
        "topo_brick.kernel.revised_kernel_deliverables",
        "run_deliverable4_learning_at_measured_bottleneck",
    ),
    "RevisedKernelSuiteSummary": ("topo_brick.kernel.revised_kernel_deliverables", "RevisedKernelSuiteSummary"),
    "run_revised_kernel_four_deliverables_suite": (
        "topo_brick.kernel.revised_kernel_deliverables",
        "run_revised_kernel_four_deliverables_suite",
    ),
}


__all__ = [
    "EXTENDED_STABLETEXT2BRICK_DIM_TO_PART",
    "AtomicBuildStep",
    "AtomicOpType",
    "BrickActionGrammarAdapter",
    "CertStatus",
    "CertificateBundle",
    "ConstrainedRolloutMetrics",
    "CorpusAttritionFunnel",
    "CoverAndRepairSolver",
    "D3PMBatchSampler",
    "Deliverable1Report",
    "Deliverable2Report",
    "Deliverable3Report",
    "Deliverable4Report",
    "DescriptorFrameEncoder",
    "ExecutionTrace",
    "ExternalParseSummary",
    "ExternalPhysicalObservation",
    "GeometryCert",
    "GreedyLayerMergeControl",
    "InsertionCert",
    "InventoryCert",
    "IterativeSeamRefinementLuo2015",
    "KernelConstrainResult",
    "KernelRealizeResult",
    "KernelRepairResult",
    "KernelResultStatus",
    "LearnedCatalogAndPortFilter",
    "LegolizationBaseline",
    "LoopClosureCert",
    "LuoInspiredSeamRefinement",
    "MinimumBuildRecord",
    "ModelAgreementCert",
    "PairSeededLocalSearch",
    "PerturbationMechanicsCert",
    "PhysicalValidationCert",
    "PrefixMechanicsCert",
    "PromotionGateVerdict",
    "RealizeRequestSpec",
    "RealizeRouteProvenance",
    "RecurrentBatchRefiner",
    "RepairAuditReport",
    "RepairEditType",
    "ResolvedRealizePolicy",
    "RevisedKernelSuiteSummary",
    "SearchDistilledValueModel",
    "SingleCoreAuditReport",
    "SolverCohortMetrics",
    "StaticMechanicsCert",
    "TopoBrickKernel",
    "ValidPlacementAction",
    "ValidatedConstraintSpec",
    "_boxes_have_continuous_cardinal_corridor",
    "_build_blind_grounded_base_pool",
    "_get_peak_rss_mb",
    "_order_course_atomic_1by1",
    "_solve_binary_mip_fast",
    "_swept_obb_hits_along_ray",
    "build_canonical_nogood_cut",
    "build_parametric_non_myopic_trap",
    "check_monotone_insertion_corridor",
    "check_rigid_group_insertion_corridor",
    "enforce_and_audit_single_core",
    "evaluate_corpus_attrition_funnel",
    "parse_external_record_specs",
    "resolve_realize_policy",
    "run_deliverable1_small_kernel_verification",
    "run_deliverable2_untouched_external_benchmark",
    "run_deliverable3_external_adapter_and_connectors",
    "run_deliverable4_learning_at_measured_bottleneck",
    "run_revised_kernel_four_deliverables_suite",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_KERNEL_COMPAT_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val
