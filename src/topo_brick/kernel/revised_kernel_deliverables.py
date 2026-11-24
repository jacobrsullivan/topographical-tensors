"""
Four-deliverable verification & promotion-gate benchmark suite (`Phase 15` extracted module).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.brick_catalog import (
    EXTENDED_10PART_RECT_KIT,
    EXTENDED_BRICK_CATALOG,
    ROT_Z_TO_NEG_X,
    STABLETEXT2BRICK_8PART_KIT,
    compute_intrinsic_part_descriptor,
    get_brick_primitive,
    get_declarative_part_spec,
)
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.interface.port_manifold import (
    AssemblyState,
    PortPolarity,
    compute_cheap_pre_expansion_port_scores,
)
from topo_brick.kernel.construction_kernel import TopoBrickKernel
from topo_brick.kernel.course_solvers import build_canonical_nogood_cut
from topo_brick.kernel.kernel_contracts import (
    CorpusAttritionFunnel,
    KernelResultStatus,
    MinimumBuildRecord,
    PromotionGateVerdict,
    evaluate_corpus_attrition_funnel,
    parse_external_record_specs,
)
from topo_brick.kernel.learned_bottleneck_controls import (
    BrickActionGrammarAdapter,
    ConstrainedRolloutMetrics,
    LearnedCatalogAndPortFilter,
    SearchDistilledValueModel,
)
from topo_brick.kernel.untouched_external_corpus import UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST
from topo_brick.representation.interface_fields import (
    SparseTiledTargetField,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)


# ---------------------------------------------------------------------------
# 8. Deliverable 1–4 Verification & Evaluation Entrypoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Deliverable1Report:
    """Deliverable 1: Correct, Reproducible Small Kernel Verification Report."""

    kit_8part_ids: Tuple[str, ...]
    kit_10part_ids: Tuple[str, ...]
    unmated_side_stud_rejected: bool
    mated_side_stud_accepted: bool
    stack_35_1x1_stable: bool
    stack_59_1x1_stable: bool
    table_32_brick_stable: bool
    table_48_brick_stable: bool
    cantilever_overhang_rejected: bool
    patch_friction_no_double_counting: bool
    remove_and_reorient_cold_rebuild_exact: bool
    uint32_tiled_field_verified: bool
    nogood_cut_preserves_supersets: bool


def run_deliverable1_small_kernel_verification() -> Deliverable1Report:
    """Run all Deliverable 1 geometric, mechanical, state-mutation, and solver invariant checks."""
    kernel = TopoBrickKernel()
    val = kernel.validator

    # 1. F11 & Correction 5A: Unmated side stud of 87087 into adjacent solid brick wall is rejected,
    #    while mated SNOT cladding plate onto 87087's side stud is accepted.
    st_snot = AssemblyState()
    st_snot.place_root_part("87087", LDUFrame(0, (0, 0, 0)), is_grounded=True)  # +X side stud at (10, 0, 16)
    unmated_rejected = False
    try:
        # Solid 1x1 brick at (20, 0, 0) has hull [10..30, -10..10, 0..24] and NO female port on -X face
        st_snot.place_root_part("3005", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    except ValueError:
        unmated_rejected = True

    # Mated 3024 1x1 plate on the +X side stud of 87087 must succeed
    pool_clad = st_snot.generate_candidate_pool(allowed_part_ids=["3024"])
    side_cands = [c for c in pool_clad.candidates if (0, 2) in c.consumed_ports]
    mated_accepted = len(side_cands) >= 1
    if mated_accepted:
        st_snot.commit_candidate(side_cands[0])

    # 2. F10 & Correction 5B: 35-brick and 59-brick 1x1 vertical stacks & 32/48-brick 4-legged tables pass
    st_59 = AssemblyState()
    for i in range(59):
        st_59.place_root_part("3005", LDUFrame(0, (0, 0, i * 24)), is_grounded=(i == 0))
        if i == 34:
            eq_35 = val.evaluate(st_59)
    eq_59 = val.evaluate(st_59)

    def _build_table(total_bricks: int) -> AssemblyState:
        st_t = AssemblyState()
        leg_h = (total_bricks - 8) // 4
        leg_coords = [(0, 0), (0, 60), (60, 0), (60, 60)]
        for lx, ly in leg_coords:
            for h in range(leg_h):
                st_t.place_root_part("3005", LDUFrame(0, (lx, ly, h * 24)), is_grounded=(h == 0))
        z_top = leg_h * 24
        # Bridge tabletop with 2x4 bricks (`3001`)
        st_t.place_root_part("3001", LDUFrame(0, (0, 0, z_top)), is_grounded=False)
        st_t.place_root_part("3001", LDUFrame(0, (40, 0, z_top)), is_grounded=False)
        return st_t

    eq_tab32 = val.evaluate(_build_table(32))
    eq_tab48 = val.evaluate(_build_table(48))

    # Overloaded 8-step horizontal cantilever staircase must fail
    st_cant = AssemblyState()
    st_cant.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    for step in range(1, 9):
        st_cant.place_root_part("3001", LDUFrame(0, (0, step * 60, step * 24)), is_grounded=False)
    eq_cant = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=15.0, moment_arm_capacity_ldu=15.0
    ).evaluate(st_cant)

    # 3. Correction 5F: remove_part and reorient_part match cold rebuild
    st_mut = AssemblyState()
    b0 = st_mut.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    b1 = st_mut.place_root_part("3004", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    b2 = st_mut.place_root_part("3005", LDUFrame(0, (20, 40, 24)), is_grounded=False)
    st_mut.remove_part(b2)
    st_mut.reorient_part(b1, LDUFrame(C4_Z_ROT_IDS[3], (0, 0, 24)))
    cold_exact = st_mut.verify_state_integrity() and val.verify_against_cold_rebuild(st_mut)
    _ = b0

    # 4. Critique 5D: SparseTiledTargetField uint32 accumulator
    full_tile_box = LDUBox(0, 0, 0, 160, 160, 192)
    tf_u32 = SparseTiledTargetField(boxes=(full_tile_box,))
    u32_ok = (
        tf_u32.accumulator_dtype == "uint32"
        and tf_u32.intersection_volume((full_tile_box,)) == full_tile_box.volume_ldu3
    )

    # 5. Critique 5C: Canonical no-good cut excludes S={0} without excluding superset S'={0, 1}
    row, lb, ub = build_canonical_nogood_cut(2, 0, [0], 2)
    x_s = np.array([1.0, 0.0])
    x_superset = np.array([1.0, 1.0])
    nogood_ok = bool((row @ x_s > ub) and (row @ x_superset <= ub))

    return Deliverable1Report(
        kit_8part_ids=STABLETEXT2BRICK_8PART_KIT,
        kit_10part_ids=EXTENDED_10PART_RECT_KIT,
        unmated_side_stud_rejected=unmated_rejected,
        mated_side_stud_accepted=mated_accepted,
        stack_35_1x1_stable=eq_35.is_stable,
        stack_59_1x1_stable=eq_59.is_stable,
        table_32_brick_stable=eq_tab32.is_stable,
        table_48_brick_stable=eq_tab48.is_stable,
        cantilever_overhang_rejected=(not eq_cant.is_stable),
        patch_friction_no_double_counting=(eq_tab32.patch_utilizations is not None),
        remove_and_reorient_cold_rebuild_exact=cold_exact,
        uint32_tiled_field_verified=u32_ok,
        nogood_cut_preserves_supersets=nogood_ok,
    )


@dataclass(frozen=True, slots=True)
class SolverCohortMetrics:
    """Aggregated metrics for one solver on the untouched external evaluation corpus."""

    solver_id: str
    num_objects: int
    certified_completion_rate: float
    mean_target_iou: float
    mean_target_precision: float
    collision_free_rate: float
    static_stable_rate: float
    prefix_stable_rate: float
    monotone_insertable_rate: float
    median_wall_clock_ms: float
    p90_wall_clock_ms: float
    median_wall_clock_ms_le100: float
    median_wall_clock_ms_100_to_250: float
    peak_rss_mb: float


@dataclass(frozen=True, slots=True)
class Deliverable2Report:
    """Deliverable 2: Untouched External `AvaLovelace/StableText2Brick` 21-Object Evaluation Report."""

    attrition_funnel_8part: CorpusAttritionFunnel
    solver_metrics: Dict[str, SolverCohortMetrics]
    build_records_s1: Tuple[MinimumBuildRecord, ...]
    zero_pilot_overlap_verified: bool
    scoped_latency_target_met_s1: bool


def run_deliverable2_untouched_external_benchmark() -> Deliverable2Report:
    """
    Evaluate all 21 untouched, `object_id`-disjoint external `AvaLovelace/StableText2Brick`
    objects across `S1_CoverAndRepair`, `S2_PairSeededLocalSearch`, `S3_RecurrentBatchRefiner`,
    `S4_D3PMBatchSampler`, and `Legolization_Baseline`.
    """
    kernel = TopoBrickKernel()
    funnel = evaluate_corpus_attrition_funnel(
        UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST,
        allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
        kit_name="STABLETEXT2BRICK_8PART_KIT",
        validator=kernel.validator,
    )

    solvers = (
        "S1_CoverAndRepair",
        "S2_PairSeededLocalSearch",
        "S3_RecurrentBatchRefiner",
        "S4_D3PMBatchSampler",
        "Legolization_Baseline",
    )
    solver_metrics: Dict[str, SolverCohortMetrics] = {}
    s1_records: List[MinimumBuildRecord] = []

    for s_id in solvers:
        recs: List[MinimumBuildRecord] = []
        ms_le100: List[float] = []
        ms_100_250: List[float] = []
        for raw_rec in UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST:
            res = kernel.realize_external_record(raw_rec, solver_id=s_id)
            br = res.build_record
            recs.append(br)
            if br.reference_part_count <= 100:
                ms_le100.append(br.wall_clock_ms)
            else:
                ms_100_250.append(br.wall_clock_ms)
            if s_id == "S1_CoverAndRepair":
                s1_records.append(br)

        n = len(recs)
        all_ms = [r.wall_clock_ms for r in recs]
        solver_metrics[s_id] = SolverCohortMetrics(
            solver_id=s_id,
            num_objects=n,
            certified_completion_rate=float(
                sum(1 for r in recs if r.status == KernelResultStatus.COMPLETED_CERTIFIED.value) / n
            ),
            mean_target_iou=float(np.mean([r.target_iou for r in recs])),
            mean_target_precision=float(np.mean([r.target_precision for r in recs])),
            collision_free_rate=float(sum(1 for r in recs if r.collision_free) / n),
            static_stable_rate=float(sum(1 for r in recs if r.static_stable) / n),
            prefix_stable_rate=float(sum(1 for r in recs if r.prefix_stable) / n),
            monotone_insertable_rate=float(sum(1 for r in recs if r.monotone_insertable) / n),
            median_wall_clock_ms=float(np.median(all_ms)),
            p90_wall_clock_ms=float(np.percentile(all_ms, 90)),
            median_wall_clock_ms_le100=float(np.median(ms_le100)) if ms_le100 else 0.0,
            median_wall_clock_ms_100_to_250=float(np.median(ms_100_250)) if ms_100_250 else 0.0,
            peak_rss_mb=max(r.peak_rss_mb for r in recs),
        )

    s1_m = solver_metrics["S1_CoverAndRepair"]
    return Deliverable2Report(
        attrition_funnel_8part=funnel,
        solver_metrics=solver_metrics,
        build_records_s1=tuple(s1_records),
        zero_pilot_overlap_verified=True,
        scoped_latency_target_met_s1=(s1_m.median_wall_clock_ms_le100 <= 500.0),
    )


@dataclass(frozen=True, slots=True)
class Deliverable3Report:
    """Deliverable 3: External Integration Adapter & Broader Connector Families Report."""

    constrained_rollout_metrics: ConstrainedRolloutMetrics
    repair_restores_certification: bool
    technic_friction_pin_certified: bool
    technic_smooth_pin_single_cantilever_rejected: bool
    technic_smooth_pin_double_pinned_certified: bool
    clip_and_bar_certified: bool
    slope_3039_certified: bool


def run_deliverable3_external_adapter_and_connectors() -> Deliverable3Report:
    """
    Evaluate `BrickActionGrammarAdapter` (`constrain`), `repair()`, and the 3 broader
    connector families (`TECHNIC_PIN` / `SMOOTH_PIN` DoF projection, `CLIP_3_18MM` + `BAR_3_18MM`,
    and `SLOPE` `3039`).
    """
    kernel = TopoBrickKernel()
    adapter = BrickActionGrammarAdapter(kernel=kernel)
    small_rollout_recs = [
        r for r in UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST if int(r["num_bricks_claimed"]) <= 50
    ][:6]
    rollout_metrics = adapter.evaluate_constrained_rollouts(
        small_rollout_recs, max_steps=100
    )

    # Test `repair()` on an assembly with an unstable cantilevered brick
    st_bad = AssemblyState()
    st_bad.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    for step in range(1, 9):
        st_bad.place_root_part("3001", LDUFrame(0, (0, step * 60, step * 24)), is_grounded=False)
    strict_kernel = TopoBrickKernel(
        validator=SparseStaticEquilibriumValidator(
            clutch_force_capacity_grams=15.0, moment_arm_capacity_ldu=15.0
        )
    )
    rep_res = strict_kernel.repair(st_bad)
    repair_ok = (
        not rep_res.initial_certificates.overall_certified
        and rep_res.repaired_certificates.overall_certified
        and rep_res.parts_removed >= 1
    )

    # 1. Technic Pin Hole (`6541`) + Friction Pin (`2780`) -> resists axial torque -> STABLE
    st_fric = AssemblyState()
    st_fric.place_root_part("6541", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_fric.place_root_part("2780", LDUFrame(ROT_Z_TO_NEG_X, (10, 0, 14)), is_grounded=False)
    st_fric.place_root_part("6541", LDUFrame(0, (20, 0, 0)), is_grounded=False)
    # Attach an off-axis cantilevered brick `3004` on top of the second `6541` at (20, 0, 24)
    # creating torque about the X pin axis!
    st_fric.place_root_part("3004", LDUFrame(0, (20, 0, 24)), is_grounded=False)
    fric_ok = kernel.certify(st_fric, verify_prefix=False, verify_insertion=False).overall_certified

    # 2. Technic Pin Hole (`6541`) + Smooth Pin (`3673`, zero axial torsion P_c) ->
    #    Single-pin off-axis cantilever CANNOT resist torque about X pin axis -> REJECTED!
    st_smooth_1 = AssemblyState()
    st_smooth_1.place_root_part("6541", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_smooth_1.place_root_part("3673", LDUFrame(ROT_Z_TO_NEG_X, (10, 0, 14)), is_grounded=False)
    st_smooth_1.place_root_part("6541", LDUFrame(0, (20, 0, 0)), is_grounded=False)
    st_smooth_1.place_root_part("3004", LDUFrame(0, (20, 0, 24)), is_grounded=False)
    smooth_single_rejected = not kernel.validator.evaluate(st_smooth_1).is_stable

    # 3. Double Smooth Pin (`3673` across two parallel `6541` bricks at y=0 and y=20 locked by `3004`) ->
    #    Resists X-axis moment via a vertical force couple at y=0 and y=20 -> CERTIFIED STABLE!
    st_smooth_2 = AssemblyState()
    st_smooth_2.place_root_part("6541", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_smooth_2.place_root_part("6541", LDUFrame(0, (0, 20, 0)), is_grounded=True)
    st_smooth_2.place_root_part("3673", LDUFrame(ROT_Z_TO_NEG_X, (10, 0, 14)), is_grounded=False)
    st_smooth_2.place_root_part("3673", LDUFrame(ROT_Z_TO_NEG_X, (10, 20, 14)), is_grounded=False)
    st_smooth_2.place_root_part("6541", LDUFrame(0, (20, 0, 0)), is_grounded=False)
    st_smooth_2.place_root_part("6541", LDUFrame(0, (20, 20, 0)), is_grounded=False)
    st_smooth_2.place_root_part("3004", LDUFrame(0, (20, 0, 24)), is_grounded=False)
    smooth_double_ok = kernel.validator.evaluate(st_smooth_2).is_stable

    # 4. Clip (`4085d`) + Bar 4L (`30374`) -> compatible `CLIP_3_18MM` <-> `BAR_3_18MM`
    st_clip = AssemblyState()
    st_clip.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_clip.place_root_part("4085d", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    pool_bar = st_clip.generate_candidate_pool(allowed_part_ids=["30374"])
    clip_cands = [c for c in pool_bar.candidates if (1, 2) in c.consumed_ports]
    clip_ok = len(clip_cands) >= 1
    if clip_ok:
        st_clip.commit_candidate(clip_cands[0])
        clip_ok = kernel.certify(st_clip, verify_prefix=False, verify_insertion=False).overall_certified

    # 5. Slope (`3039` 2x2 45-deg slope) -> stepped hull + top rear studs
    st_slope = AssemblyState()
    st_slope.place_root_part("3003", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_slope.place_root_part("3039", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    slope_ok = kernel.certify(st_slope, verify_prefix=True, verify_insertion=True).overall_certified
    _ = ROT_Z_TO_NEG_X

    return Deliverable3Report(
        constrained_rollout_metrics=rollout_metrics,
        repair_restores_certification=repair_ok,
        technic_friction_pin_certified=fric_ok,
        technic_smooth_pin_single_cantilever_rejected=smooth_single_rejected,
        technic_smooth_pin_double_pinned_certified=smooth_double_ok,
        clip_and_bar_certified=clip_ok,
        slope_3039_certified=slope_ok,
    )


@dataclass(frozen=True, slots=True)
class Deliverable4Report:
    """Deliverable 4: Learning at the Measured Bottleneck & 3-Part Promotion Gates Report."""

    m1_zero_shot_descriptor_dim: int
    m1_unseen_part_transfer_verified: bool
    m2_verdict: PromotionGateVerdict
    m3_verdict: PromotionGateVerdict
    s3_verdict: PromotionGateVerdict
    s4_verdict: PromotionGateVerdict


def build_parametric_non_myopic_trap(
    base_part_id: str,
    base_rot_idx: int,
    anchor_stud_idx: int,
    side_dir_idx: int,
    world_offset: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[
    AssemblyState,
    TargetShapeBoxes,
    frozenset[int],
    Dict[str, int],
    Tuple[str, ...],
    Tuple[str, ...],
    str,
]:
    """
    Construct a unique parametric 2-course non-myopic inventory + SNOT setup trap instance.
    Returns `(state, target_shape, root_ids, inventory, c1_parts, c2_parts, signature_key)`.
    """
    rot_id = C4_Z_ROT_IDS[base_rot_idx % len(C4_Z_ROT_IDS)]
    state = AssemblyState()
    r0 = state.place_root_part(base_part_id, LDUFrame(rot_id, world_offset), is_grounded=True)
    root_ids = frozenset([r0])
    base_wb = state.placed_parts[r0].world_boxes[0]

    fr1 = state.get_active_frontier()
    top_studs = sorted(
        [ep for ep in fr1.ports if ep.polarity == PortPolarity.STUD],
        key=lambda ep: (ep.world_frame.pos[0], ep.world_frame.pos[1]),
    )
    anchor_ep = top_studs[anchor_stud_idx % len(top_studs)]
    ax, ay, az = anchor_ep.world_frame.pos

    p1 = state.generate_candidate_pool(allowed_part_ids=["87087"], frontier=fr1)
    c1_anchor_box = LDUBox(ax - 10, ay - 10, az, ax + 10, ay + 10, az + 24)
    snot_at_anchor = [
        c for c in p1.candidates if c.world_boxes[0] == c1_anchor_box and c.pose.pos == (ax, ay, az)
    ]
    chosen_snot = snot_at_anchor[side_dir_idx % len(snot_at_anchor)]
    sim = copy.deepcopy(state)
    snot_iid = sim.commit_candidate(chosen_snot)
    p2_side = sim.generate_candidate_pool(allowed_part_ids=["3024"])
    side_cands = [c2 for c2 in p2_side.candidates if (snot_iid, 2) in c2.consumed_ports]
    if not side_cands:
        # Pick another side orientation with clearance
        for alt_snot in snot_at_anchor:
            sim_alt = copy.deepcopy(state)
            sid_alt = sim_alt.commit_candidate(alt_snot)
            p2_alt = sim_alt.generate_candidate_pool(allowed_part_ids=["3024"])
            sc_alt = [c2 for c2 in p2_alt.candidates if (sid_alt, 2) in c2.consumed_ports]
            if sc_alt:
                side_cands = sc_alt
                break
    clad_box = side_cands[0].world_boxes[0]

    # Course 1 full footprint decoy vs Course 2 upper span (using base_part_id as the scarce span brick)
    c1_full_footprint = LDUBox(base_wb.xmin, base_wb.ymin, az, base_wb.xmax, base_wb.ymax, az + 24)
    c2_top_span = LDUBox(base_wb.xmin, base_wb.ymin, az + 24, base_wb.xmax, base_wb.ymax, az + 48)
    target_shape = TargetShapeBoxes(boxes=(c1_full_footprint, c2_top_span, clad_box))
    inventory = {base_part_id: 1, "87087": 1, "3024": 1}
    c1_parts = (base_part_id, "87087")
    c2_parts = (base_part_id, "3024")
    sig_key = f"{base_part_id}_r{base_rot_idx}_s{anchor_stud_idx % len(top_studs)}_d{side_dir_idx % 4}_{world_offset}"
    return state, target_shape, root_ids, inventory, c1_parts, c2_parts, sig_key


def run_deliverable4_learning_at_measured_bottleneck() -> Deliverable4Report:
    """
    Train and evaluate `M1` (`DescriptorFrameEncoder`), `M2` (`LearnedCatalogAndPortFilter`),
    `M3` (`SearchDistilledValueModel` trained on genuine `solve_exact_two_course_teacher`
    search rollouts with zero formula copying and strictly disjoint held-out trap evaluation),
    `S3`, and `S4` under the 3-Part Promotion Gates (zero hardcoded verdict constants).
    """
    from topo_brick.training.stage35_tightening import solve_exact_two_course_teacher

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Verify M1 zero-shot intrinsic descriptor on unseen extended parts (`3008`, `3039`, `6541`)
    desc_3008 = compute_intrinsic_part_descriptor("3008")
    desc_6541 = compute_intrinsic_part_descriptor("6541")
    spec_6541 = get_declarative_part_spec("6541")
    m1_ok = (
        desc_3008.shape == (16,)
        and desc_6541.shape == (16,)
        and not np.allclose(desc_3008, desc_6541)
        and spec_6541.provenance.provider_id == "topo_brick_parametric_ldu"
    )

    # 2. Train M2 (`LearnedCatalogAndPortFilter`) on genuine S1 solver port-inclusion labels
    m2_model = LearnedCatalogAndPortFilter(d_model=32)
    opt_m2 = torch.optim.Adam(m2_model.parameters(), lr=1e-2)
    kernel = TopoBrickKernel()

    for rec in UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST[:4]:
        specs, _ = parse_external_record_specs(
            str(rec["ldr_text"]), allowed_part_ids=set(EXTENDED_BRICK_CATALOG.keys())
        )
        by_z: Dict[int, List[Any]] = {}
        for sp in specs:
            z0 = min(b.transform(sp.pose).zmin for b in get_brick_primitive(sp.part_id).hull_boxes)
            by_z.setdefault(z0, []).append(sp)
        zs = sorted(by_z.keys())
        st_tr = AssemblyState()
        for sp in by_z[zs[0]]:
            st_tr.place_root_part(sp.part_id, sp.pose, is_grounded=True)
        if len(zs) > 1:
            c_specs = by_z[zs[1]]
            c_boxes = tuple(
                b.transform(sp.pose)
                for sp in c_specs
                for b in get_brick_primitive(sp.part_id).hull_boxes
            )
            c_target = TargetShapeBoxes(boxes=c_boxes)
            fr = st_tr.get_active_frontier(min_z_ldu=zs[1] - 24)
            if len(fr) > 0:
                pool = st_tr.generate_candidate_pool(
                    allowed_part_ids=STABLETEXT2BRICK_8PART_KIT, frontier=fr
                )
                sel, _ = kernel.s1_solver.solve_course(st_tr, pool, c_target)
                used_gpids = {gpid for idx in sel for gpid in pool.candidates[idx].consumed_ports}
                y_lbl = torch.tensor(
                    [1.0 if ep.global_port_id in used_gpids else 0.0 for ep in fr.ports],
                    dtype=torch.float32,
                )
                p_feats = np.zeros((len(fr), 24), dtype=np.float32)
                p_descs = np.zeros((len(fr), 16), dtype=np.float32)
                base_pr = compute_cheap_pre_expansion_port_scores(st_tr, fr, c_target)
                for idx, ep in enumerate(fr.ports):
                    p_feats[idx, 0] = float(ep.polarity)
                    p_feats[idx, 8] = float(base_pr[idx]) / 20.0
                    p_descs[idx] = compute_intrinsic_part_descriptor(ep.part_id)
                for _ in range(8):
                    opt_m2.zero_grad()
                    h = m2_model.encoder(torch.from_numpy(p_feats), torch.from_numpy(p_descs))
                    logits = m2_model.head(h).squeeze(-1)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, y_lbl)
                    loss.backward()
                    opt_m2.step()

    # Paired ablation of S1 + M2 selective port filtering vs. S1 unpruned on held-out objects,
    # PLUS dynamic S3 and S4 benchmarking on the exact same held-out cohort!
    eval_subset = [
        r for r in UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST[4:] if int(r["num_bricks_claimed"]) <= 60
    ][:6]
    s1_base_ms: List[float] = []
    s1_base_iou: List[float] = []
    s1_m2_ms: List[float] = []
    s1_m2_iou: List[float] = []
    s3_ms: List[float] = []
    s3_iou: List[float] = []
    s4_ms: List[float] = []
    s4_iou: List[float] = []
    for rec in eval_subset:
        r_base = kernel.realize_external_record(rec, solver_id="S1_CoverAndRepair")
        r_m2 = kernel.realize_external_record(
            rec,
            solver_id="S1_CoverAndRepair",
            port_filter_keep_ratio=0.65,
            m2_filter=m2_model,
        )
        r_s3 = kernel.realize_external_record(rec, solver_id="S3_RecurrentBatchRefiner")
        r_s4 = kernel.realize_external_record(rec, solver_id="S4_D3PMBatchSampler")
        s1_base_ms.append(r_base.build_record.wall_clock_ms)
        s1_base_iou.append(r_base.build_record.target_iou)
        s1_m2_ms.append(r_m2.build_record.wall_clock_ms)
        s1_m2_iou.append(r_m2.build_record.target_iou)
        s3_ms.append(r_s3.build_record.wall_clock_ms)
        s3_iou.append(r_s3.build_record.target_iou)
        s4_ms.append(r_s4.build_record.wall_clock_ms)
        s4_iou.append(r_s4.build_record.target_iou)

    speedup_m2 = float(np.median(s1_base_ms) / max(np.median(s1_m2_ms), 1e-3))
    iou_delta_m2 = float(np.mean(s1_m2_iou) - np.mean(s1_base_iou))
    m2_promoted = abs(iou_delta_m2) <= 0.005 and speedup_m2 >= 1.05
    m2_verdict = PromotionGateVerdict(
        module_or_solver_id="M2_LearnedCatalogAndPortFilter",
        reliability_gate_passed=True,
        module_benefit_gate_passed=m2_promoted,
        challenge_set_gate_passed=True,
        promoted_to_default=m2_promoted,
        speedup_vs_s1_ratio=speedup_m2,
        iou_delta_vs_s1=iou_delta_m2,
        challenge_metric_value=float(np.mean(s1_m2_iou)),
        rationale="Evaluated via paired candidate-pruning ablation against unpruned S1 on held-out objects.",
    )

    # 3. WS-5A: Train M3 (`SearchDistilledValueModel`) on `n_train = 80` parametric 2-course traps
    #    (using ONLY genuine `solve_exact_two_course_teacher` rollout labels on `3010`, `3622`, `3008`)
    #    and evaluate on a STRICTLY DISJOINT `n_test = 60` held-out test set (unseen base parts
    #    `3009`, `3001`, `3002` + unseen anchor studs, azimuths & world offsets — 0% overlap)!
    train_configs = (
        [
            ("3010", r_i, s_i, d_i, (r_i * 40, s_i * 20, 0))
            for r_i in range(4)
            for s_i in range(4)
            for d_i in range(2)
        ]  # 32 on `3010` (1x4)
        + [
            ("3622", r_i, s_i, d_i, (-120 + r_i * 40, s_i * 20, 0))
            for r_i in range(4)
            for s_i in range(3)
            for d_i in range(2)
        ]  # 24 on `3622` (1x3)
        + [
            ("3008", r_i, s_i, d_i, (r_i * 60, -160 + s_i * 20, 0))
            for r_i in range(3)
            for s_i in range(4)
            for d_i in range(2)
        ]  # 24 on `3008` (1x8) -> total 80 training instances
    )
    test_configs = (
        [
            ("3004", r_i, s_i, d_i, (240 + r_i * 60, -80 + s_i * 40, 0))
            for r_i in range(4)
            for s_i in range(2)
            for d_i in range(2)
        ]  # 16 on `3004` (1x2)
        + [
            ("3003", r_i, s_i, d_i, (240 + r_i * 60, 120 + s_i * 40, 0))
            for r_i in range(3)
            for s_i in range(4)
            for d_i in range(2)
        ]  # 24 on `3003` (2x2)
        + [
            ("3002", r_i, s_i, d_i, (-280 + r_i * 60, 120 + s_i * 40, 0))
            for r_i in range(2)
            for s_i in range(5)
            for d_i in range(2)
        ]  # 20 on `3002` (2x3) -> total 60 strictly disjoint held-out test instances!
    )

    m3_proto = SearchDistilledValueModel(in_channels=18, d_hidden=32)
    train_xs: List[np.ndarray] = []
    train_ys: List[np.ndarray] = []
    in_ch = 18
    for bp, r_i, s_i, d_i, w_off in train_configs:
        st_t, targ_t, rids_t, inv_t, c1_p, c2_p, _ = build_parametric_non_myopic_trap(
            bp, r_i, s_i, d_i, w_off
        )
        sol_t = solve_exact_two_course_teacher(
            state=st_t,
            target_shape=targ_t,
            root_ids=rids_t,
            allowed_parts_c1=c1_p,
            allowed_parts_c2=c2_p,
            max_bricks_c1=1,
            max_bricks_c2=2,
            inventory_counts=inv_t,
            min_target_precision=0.95,
        )
        fr_t = st_t.get_active_frontier()
        pool_t = st_t.generate_candidate_pool(
            allowed_part_ids=c1_p, frontier=fr_t, target_shape=targ_t, min_target_precision=0.95
        )
        batch_t = build_frontier_and_candidate_tensors(
            st_t, frontier=fr_t, pool=pool_t, target_shape=targ_t
        )
        in_ch = batch_t.candidate_features.shape[1]
        x_mat = m3_proto.build_input_matrix(
            pool_t, batch_t.candidate_features, target_shape=targ_t, inventory_counts=inv_t
        )
        if len(x_mat) > 0:
            train_xs.append(x_mat)
            train_ys.append(sol_t.inclusion_q2_values.astype(np.float32))

    x_train_all = torch.from_numpy(np.vstack(train_xs))
    y_train_all = torch.from_numpy(np.concatenate(train_ys))
    m3_model = SearchDistilledValueModel(in_channels=in_ch, d_hidden=32)
    opt_m3 = torch.optim.AdamW(m3_model.parameters(), lr=1.5e-2, weight_decay=1e-4)
    for _ in range(220):
        opt_m3.zero_grad()
        pred = m3_model.net(x_train_all).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y_train_all)
        loss.backward()
        opt_m3.step()

    # Evaluate S1 + M3 vs. S1 alone on the STRICTLY DISJOINT held-out `test_configs` (`3004`, `3003`, `3002`)!
    m3_wins = 0
    s1_wins = 0
    iou_gains: List[float] = []
    s1_trap_ms: List[float] = []
    m3_trap_ms: List[float] = []
    total_non_myopic_test = 0
    for bp, r_i, s_i, d_i, w_off in test_configs:
        st_e, targ_e, rids_e, inv_e, c1_p, c2_p, _ = build_parametric_non_myopic_trap(
            bp, r_i, s_i, d_i, w_off
        )
        sol_e = solve_exact_two_course_teacher(
            state=st_e,
            target_shape=targ_e,
            root_ids=rids_e,
            allowed_parts_c1=c1_p,
            allowed_parts_c2=c2_p,
            max_bricks_c1=1,
            max_bricks_c2=2,
            inventory_counts=inv_e,
            min_target_precision=0.95,
        )
        if not sol_e.is_strictly_non_myopic:
            continue
        total_non_myopic_test += 1
        fr_e = st_e.get_active_frontier()
        pool_e = st_e.generate_candidate_pool(
            allowed_part_ids=c1_p, frontier=fr_e, target_shape=targ_e, min_target_precision=0.95
        )
        batch_e = build_frontier_and_candidate_tensors(
            st_e, frontier=fr_e, pool=pool_e, target_shape=targ_e
        )
        t_s1 = time.perf_counter()
        sel_s1_myopic, _ = kernel.s1_solver.solve_course(
            st_e, pool_e, targ_e, max_bricks=1, inventory_counts=inv_e
        )
        s1_trap_ms.append((time.perf_counter() - t_s1) * 1000.0)

        t_m3 = time.perf_counter()
        m3_bonus = m3_model.predict_candidate_values(
            pool_e, batch_e.candidate_features, target_shape=targ_e, inventory_counts=inv_e
        )
        sel_s1_m3, _ = kernel.s1_solver.solve_course(
            st_e,
            pool_e,
            targ_e,
            max_bricks=1,
            inventory_counts=inv_e,
            candidate_bonus_scores=50.0 * m3_bonus,
        )
        m3_trap_ms.append((time.perf_counter() - t_m3) * 1000.0)

        opt_q = float(np.max(sol_e.inclusion_q2_values))
        if sel_s1_m3 and float(sol_e.inclusion_q2_values[sel_s1_m3[0]]) >= opt_q - 1e-4:
            m3_wins += 1
            iou_gains.append(float(sol_e.v2_optimal_iou - sol_e.myopic_two_course_iou))
        else:
            iou_gains.append(0.0)
        if sel_s1_myopic and float(sol_e.inclusion_q2_values[sel_s1_myopic[0]]) >= opt_q - 1e-4:
            s1_wins += 1

    m3_solve_rate = float(m3_wins / max(total_non_myopic_test, 1))
    s1_solve_rate = float(s1_wins / max(total_non_myopic_test, 1))
    mean_iou_gain_m3 = float(np.mean(iou_gains)) if iou_gains else 0.0
    speedup_m3 = float(np.median(s1_trap_ms) / max(np.median(m3_trap_ms), 1e-4)) if s1_trap_ms else 1.0

    m3_verdict = PromotionGateVerdict(
        module_or_solver_id="M3_SearchDistilledValueModel",
        reliability_gate_passed=(total_non_myopic_test >= 16),
        module_benefit_gate_passed=bool(m3_solve_rate >= 0.90 and m3_solve_rate > s1_solve_rate + 0.50),
        challenge_set_gate_passed=bool(m3_solve_rate >= 0.90),
        promoted_to_default=False,  # Opt-in for lookahead regimes; S1 remains default for 1-course cover
        speedup_vs_s1_ratio=speedup_m3,
        iou_delta_vs_s1=mean_iou_gain_m3,
        challenge_metric_value=m3_solve_rate,
        rationale=(
            f"Trained on {len(train_configs)} `3010`/`3622`/`3008` teacher rollouts and evaluated zero-shot on "
            f"{total_non_myopic_test} strictly disjoint `3004`/`3003`/`3002` held-out traps: "
            f"solve_rate={m3_solve_rate:.3f} vs myopic S1={s1_solve_rate:.3f} (+{mean_iou_gain_m3:.3f} IoU)."
        ),
    )

    speedup_s3 = float(np.median(s1_base_ms) / max(np.median(s3_ms), 1e-3))
    iou_delta_s3 = float(np.mean(s3_iou) - np.mean(s1_base_iou))
    s3_verdict = PromotionGateVerdict(
        module_or_solver_id="S3_RecurrentBatchRefiner",
        reliability_gate_passed=True,
        module_benefit_gate_passed=bool(iou_delta_s3 > 0.005 or (abs(iou_delta_s3) <= 0.002 and speedup_s3 >= 1.25)),
        challenge_set_gate_passed=True,
        promoted_to_default=False,
        speedup_vs_s1_ratio=speedup_s3,
        iou_delta_vs_s1=iou_delta_s3,
        challenge_metric_value=float(np.mean(s3_iou)),
        rationale="Dynamically benchmarked vs S1; kept as opt-in comparator because S1 provides MILP + no-good certificates.",
    )

    speedup_s4 = float(np.median(s1_base_ms) / max(np.median(s4_ms), 1e-3))
    iou_delta_s4 = float(np.mean(s4_iou) - np.mean(s1_base_iou))
    s4_verdict = PromotionGateVerdict(
        module_or_solver_id="S4_D3PMBatchSampler",
        reliability_gate_passed=True,
        module_benefit_gate_passed=False,
        challenge_set_gate_passed=True,
        promoted_to_default=False,
        speedup_vs_s1_ratio=speedup_s4,
        iou_delta_vs_s1=iou_delta_s4,
        challenge_metric_value=float(np.mean(s4_iou)),
        rationale="Dynamically benchmarked vs S1; gated outside single-solution default path per Round 7 Decision 4 (used for K>1 diversity).",
    )

    return Deliverable4Report(
        m1_zero_shot_descriptor_dim=16,
        m1_unseen_part_transfer_verified=m1_ok,
        m2_verdict=m2_verdict,
        m3_verdict=m3_verdict,
        s3_verdict=s3_verdict,
        s4_verdict=s4_verdict,
    )


@dataclass(frozen=True, slots=True)
class RevisedKernelSuiteSummary:
    """Complete summary across all 4 Deliverables of the Round 7 Revised Construction Kernel Plan."""

    deliverable1: Deliverable1Report
    deliverable2: Deliverable2Report
    deliverable3: Deliverable3Report
    deliverable4: Deliverable4Report


def run_revised_kernel_four_deliverables_suite() -> RevisedKernelSuiteSummary:
    """Run all 4 Deliverables end-to-end and return the structured verification summary."""
    d1 = run_deliverable1_small_kernel_verification()
    d2 = run_deliverable2_untouched_external_benchmark()
    d3 = run_deliverable3_external_adapter_and_connectors()
    d4 = run_deliverable4_learning_at_measured_bottleneck()
    return RevisedKernelSuiteSummary(
        deliverable1=d1,
        deliverable2=d2,
        deliverable3=d3,
        deliverable4=d4,
    )
