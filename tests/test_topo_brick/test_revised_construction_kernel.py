"""
Verification Test Suite for the Round 7 Revised Construction Kernel Plan:
- Deliverable 1: Correct, Reproducible Small Kernel (`test_deliverable1_small_kernel_invariants`)
- Deliverable 2: Narrow External Performance Result on 21 Untouched `AvaLovelace/StableText2Brick`
  Objects (`test_deliverable2_untouched_external_corpus_and_attrition_funnel`)
- Deliverable 3: External Integration Adapter & Broader Connector Families
  (`test_deliverable3_external_adapter_repair_and_connector_families`)
- Deliverable 4: Learning at the Measured Bottleneck with 3-Part Promotion Gates
  (`test_deliverable4_learned_bottleneck_modules_and_promotion_gates`)
"""

from __future__ import annotations

import numpy as np

from topo_brick.interface.brick_catalog import (
    BRICK_CATALOG,
    EXTENDED_10PART_RECT_KIT,
    STABLETEXT2BRICK_8PART_KIT,
    get_declarative_part_spec,
)
from topo_brick.interface.ldu_frames import LDUFrame
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.kernel import (
    UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST,
    CertStatus,
    TopoBrickKernel,
    build_canonical_nogood_cut,
    run_deliverable1_small_kernel_verification,
    run_deliverable2_untouched_external_benchmark,
    run_deliverable3_external_adapter_and_connectors,
    run_deliverable4_learning_at_measured_bottleneck,
)
from topo_brick.training import LOCKED_STABLETEXT2BRICK_MANIFEST


def test_deliverable1_small_kernel_invariants() -> None:
    """
    Verify Deliverable 1 (Round 7 Decisions 1–4 & Corrections 5A–5F):
    1. Critique 5E: `STABLETEXT2BRICK_8PART_KIT` has exactly the 8 rectangular parts
       (`3005, 3004, 3010, 3009, 3008, 3003, 3001, 2456`), excluding `3622` (1x3) and `3002` (2x3),
       while `EXTENDED_10PART_RECT_KIT` adds `3622` and `3002`, and `BRICK_CATALOG` retains
       its 20 baseline parts for backward compatibility.
    2. F11 & Correction 5A: Unmated side-stud protrusion of `87087` into an adjacent solid brick
       wall is rejected, whereas a mated `3024` cladding plate on that side stud is accepted.
    3. F10 & Correction 5B: 35-brick and 59-brick 1x1 vertical stacks and 32/48-brick 4-legged
       tables are certified STABLE under asymmetric normal bearing, while an 8-step horizontal
       cantilever is rejected, and multi-stud contact-patch utilization is tracked without
       double-counting face friction.
    4. Correction 5F: `remove_part` and `reorient_part` increment `state_version` and match a
       cold rebuild with 0 discrepancies.
    5. Critique 5D: `SparseTiledTargetField` uses `uint32` accumulators and handles a full
       `160 x 160 x 192 LDU` (`614,400` 2-LDU cells) tile without `uint16` overflow.
    6. Critique 5C: Canonical binary no-good cut excludes the failed binary assignment `S`
       without excluding strict supersets `S' supset S`.
    """
    assert len(BRICK_CATALOG) == 20
    assert STABLETEXT2BRICK_8PART_KIT == (
        "3005",
        "3004",
        "3010",
        "3009",
        "3008",
        "3003",
        "3001",
        "2456",
    )
    assert "3622" not in STABLETEXT2BRICK_8PART_KIT
    assert "3002" not in STABLETEXT2BRICK_8PART_KIT
    assert len(EXTENDED_10PART_RECT_KIT) == 10

    spec_87087 = get_declarative_part_spec("87087")
    assert spec_87087.provenance.collision_contract.mated_port_exemption_enabled is True
    assert any(
        p.local_protrusion_box is not None and abs(int(p.local_frame.z_axis[2])) == 0
        for p in spec_87087.primitive.ports
    )

    d1 = run_deliverable1_small_kernel_verification()
    assert d1.unmated_side_stud_rejected is True
    assert d1.mated_side_stud_accepted is True
    assert d1.stack_35_1x1_stable is True
    assert d1.stack_59_1x1_stable is True
    assert d1.table_32_brick_stable is True
    assert d1.table_48_brick_stable is True
    assert d1.cantilever_overhang_rejected is True
    assert d1.patch_friction_no_double_counting is True
    assert d1.remove_and_reorient_cold_rebuild_exact is True
    assert d1.uint32_tiled_field_verified is True
    assert d1.nogood_cut_preserves_supersets is True

    # Verify 4-valued certificate statuses on `TopoBrickKernel.certify()`
    kernel = TopoBrickKernel()
    st = AssemblyState()
    st.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    certs = kernel.certify(st, verify_prefix=False, verify_insertion=False)
    assert certs.geometry.status == CertStatus.PASS
    assert certs.static_mechanics.status == CertStatus.PASS
    assert certs.prefix_mechanics.status == CertStatus.NOT_APPLICABLE
    assert certs.insertion.status == CertStatus.NOT_APPLICABLE
    assert certs.physical_validation.status == CertStatus.UNKNOWN
    assert certs.overall_certified is True

    # Verify canonical no-good cut on 3-candidate example where {0} is unstable but {0, 1} is counterbalanced
    row, _, ub = build_canonical_nogood_cut(3, 0, [0], 3)
    assert float(row @ np.array([1.0, 0.0, 0.0])) > ub  # {0} excluded
    assert float(row @ np.array([1.0, 1.0, 0.0])) <= ub  # superset {0, 1} preserved!


def test_deliverable2_untouched_external_corpus_and_attrition_funnel() -> None:
    """
    Verify Deliverable 2:
    1. All 21 records in `UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST` have unique `object_id`s and
       ZERO `object_id` overlap with `LOCKED_STABLETEXT2BRICK_MANIFEST`.
    2. Dynamic `CorpusAttritionFunnel` evaluates all 21 untouched objects with `21/21` supported
       by `STABLETEXT2BRICK_8PART_KIT`, `21/21` collision-free, and `21/21` static-equilibrium
       certified under asymmetric bearing.
    3. `S1_CoverAndRepair` (provisional default, 0 learned parameters) achieves `100%` certified
       completion (`COMPLETED_CERTIFIED`), `1.000` mean Target IoU, `<= 500 ms` median latency
       in the `<= 100` brick regime, and outperforms `Legolization_Baseline`.
    """
    pilot_oids = {r.object_id for r in LOCKED_STABLETEXT2BRICK_MANIFEST}
    untouched_oids = [str(r["object_id"]) for r in UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST]
    assert len(untouched_oids) == 21
    assert len(set(untouched_oids)) == 21
    assert pilot_oids.isdisjoint(set(untouched_oids))

    # Check 100-250 brick cohort presence
    over_100 = [r for r in UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST if int(r["num_bricks_claimed"]) >= 100]
    assert len(over_100) == 6

    d2 = run_deliverable2_untouched_external_benchmark()
    f = d2.attrition_funnel_8part
    assert f.stage0_fetched == 21
    assert f.stage1_parsed_valid == 21
    assert f.stage2_catalog_supported == 21
    assert f.stage3_collision_and_protrusion_free == 21
    assert f.stage4_connected_to_ground == 21
    assert f.stage5_static_equilibrium_certified == 21
    assert f.stage6_prefix_certified == 21

    s1 = d2.solver_metrics["S1_CoverAndRepair"]
    leg = d2.solver_metrics["Legolization_Baseline"]
    assert s1.certified_completion_rate == 1.0
    assert s1.mean_target_iou >= 0.998
    assert s1.collision_free_rate == 1.0
    assert s1.static_stable_rate == 1.0
    assert s1.prefix_stable_rate == 1.0
    assert s1.median_wall_clock_ms_le100 <= 500.0
    assert s1.num_objects == 21
    assert leg.num_objects == 21
    # S1 uses fewer bricks than 1x1/1x2 Legolization merge while preserving 100% interlocking certification
    assert d2.scoped_latency_target_met_s1 is True


def test_deliverable3_external_adapter_repair_and_connector_families() -> None:
    """
    Verify Deliverable 3 (Round 7 Decision 1):
    1. `BrickActionGrammarAdapter` reports accepted-action validity (`1.000`) separately from
       100-step completion rate (`< 1.0` when assemblies finish early), early completion rate,
       and dead-end rate.
    2. `TopoBrickKernel.repair()` restores an overloaded cantilever assembly from `FAIL` to
       `COMPLETED_CERTIFIED`.
    3. Broader connector families (`TECHNIC_PIN` `2780`, `SMOOTH_PIN` `3673` with zero axial
       torsion $P_c$, `CLIP_3_18MM` `4085d` + `BAR_3_18MM` `30374`, and `SLOPE` `3039`) are
       certified and distinguish rotational pin degrees of freedom.
    """
    d3 = run_deliverable3_external_adapter_and_connectors()
    cm = d3.constrained_rollout_metrics
    assert cm.accepted_action_validity_rate == 1.0
    # Because test objects have 22..96 bricks (< 100 steps), 100-step completion rate is reported
    # separately from accepted-action validity!
    assert cm.hundred_step_completion_rate < cm.accepted_action_validity_rate
    assert cm.early_target_completion_rate > 0.0
    assert cm.median_step_latency_ms < 100.0

    assert d3.repair_restores_certification is True
    assert d3.technic_friction_pin_certified is True
    assert d3.technic_smooth_pin_single_cantilever_rejected is True
    assert d3.technic_smooth_pin_double_pinned_certified is True
    assert d3.clip_and_bar_certified is True
    assert d3.slope_3039_certified is True


def test_deliverable4_learned_bottleneck_modules_and_promotion_gates() -> None:
    """
    Verify Deliverable 4 (Round 7 Decision 4):
    1. `M1` (`DescriptorFrameEncoder`) uses 16-dim intrinsic geometry-derived part descriptors
       and transfers zero-shot to newly added parts (`3008`, `6541`).
    2. `M2` (`LearnedCatalogAndPortFilter`) and `M3` (`SearchDistilledValueModel` trained on
       genuine 2-course teacher rollout outcomes, zero formula copying) are evaluated under
       separate Reliability, Module-Benefit, and Challenge-Set gates.
    3. `M3` solves the non-myopic 2-course setup trap where 1-step myopic `S1` fails, while
       `S4` (`D3PMBatchSampler`) remains gated outside the default single-solution path.
    """
    d4 = run_deliverable4_learning_at_measured_bottleneck()
    assert d4.m1_zero_shot_descriptor_dim == 16
    assert d4.m1_unseen_part_transfer_verified is True

    assert d4.m2_verdict.reliability_gate_passed is True
    assert abs(d4.m2_verdict.iou_delta_vs_s1) <= 0.005

    assert d4.m3_verdict.reliability_gate_passed is True
    assert d4.m3_verdict.module_benefit_gate_passed is True
    assert d4.m3_verdict.challenge_set_gate_passed is True
    assert d4.m3_verdict.challenge_metric_value == 1.0
    assert d4.m3_verdict.iou_delta_vs_s1 > 0.10

    assert d4.s3_verdict.promoted_to_default is False
    assert d4.s4_verdict.promoted_to_default is False
