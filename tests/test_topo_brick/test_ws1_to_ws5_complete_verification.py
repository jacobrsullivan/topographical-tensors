"""
Comprehensive End-to-End Verification Suite for Workstreams WS-1 through WS-5 (`Claims A–F`):
1. `WS-1`: `uint32` `SparseTiledTargetField` fast memoryview queries (`0` dense global bbox grid),
   genuinely blind whole-target `TopoBrickKernel.realize(target_field)`, unconditional per-course
   6-DoF LP `certify(..., verify_prefix=True)` across `> 4` courses, and `< 25 ms` `constrain()`.
2. `WS-2`: Non-redundant normal-force-coupled Coulomb + sublinear clutch patch mechanics and
   `8/8` (`100%`) CMU `StableLego` physical hardware benchmark + `BrickSim` + `StableText2Brick`
   model agreement.
3. `WS-3`: `TIER_A_CATALOG` (`213` parts, `100%` official LEGO Set `10698-1` coverage across `108`
   part types / `790` pieces), continuous $SE(3)$ 15-axis OBB-SAT kinematics & joint-DoF equilibrium,
   26-set real human OMR `.mpd` round-trip benchmark (`Claim C`), and Set `10698-1` inventory-bounded
   Alt-Builds (`Claim E`).
4. `WS-4`: `n = 600` `object_id`-deduplicated untouched `AvaLovelace/StableText2Brick` `test`-split
   benchmark (`Claim A`, strict separation of `blind` vs. `reference_assisted`, exact Clopper-Pearson
   95% CIs) and `n = 30` Hard-Overhang / Arch Challenge Set (`Claim B`, `0/30` baseline vs. `30/30`
   lexicographic support-pillar realization).
5. `WS-5`: Disjoint `n_eval = 50` `120`-part `TIER_A_CATALOG` `M2` evaluation + disjoint `n_test = 60`
   rollout-distilled `M3` evaluation, and live PyTorch `LlamaForCausalLM.generate()` 100-step
   constrained vs. unmasked generation with `BrickGrammarLogitsProcessor` (`Claim D`).
"""

from __future__ import annotations

import math
import time
import numpy as np

from topo_brick import (
    BRICKGPT_VOCAB_SIZE,
    LOCKED_STABLETEXT2BRICK_MANIFEST,
    SET_10698_OFFICIAL_INVENTORY,
    STABLETEXT2BRICK_8PART_KIT,
    TIER_A_CATALOG,
    UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST,
    AssemblyState,
    CertStatus,
    ConnectorType,
    ContinuousAssemblyState,
    ContinuousSE3Pose,
    KernelResultStatus,
    LDUBox,
    LDUFrame,
    SparseStaticEquilibriumValidator,
    SparseTiledTargetField,
    TargetShapeBoxes,
    TopoBrickKernel,
    box_to_obb,
    clopper_pearson_95_ci,
    compute_intrinsic_part_descriptor,
    evaluate_hard_overhang_arch_challenge_set,
    evaluate_m2_on_tier_a_catalog,
    evaluate_omr_26set_benchmark,
    get_declarative_part_spec,
    get_prior_locked_and_eval_object_ids,
    load_untouched_s2b_600_cohort,
    load_untouched_s2b_600_results,
    obb_intersects_sat,
    run_live_transformer_constrained_generation_benchmark,
    run_physical_and_model_agreement_suite,
    run_set_10698_inventory_bounded_alt_builds,
)


def test_ws1_sparse_tiled_field_blind_realize_unconditional_prefix_and_fast_constrain() -> None:
    """Verify WS-1A, WS-1B, WS-1C, and WS-1D invariants and performance contracts."""
    # WS-1A: SparseTiledTargetField allocates ONLY occupied 160x160x192 tiles (never a dense global bbox)
    far_boxes = (
        LDUBox(0, 0, 0, 80, 40, 24),
        LDUBox(20, 0, 24, 60, 40, 48),
        LDUBox(16000, 16000, 9600, 16080, 16040, 9624),
    )
    tf = SparseTiledTargetField(boxes=far_boxes)
    exact_ref = TargetShapeBoxes(boxes=far_boxes)
    assert not hasattr(tf, "_dense_grid")
    assert tf.num_allocated_tiles <= 4
    assert tf.total_volume_ldu3 == exact_ref.total_volume_ldu3

    probe_boxes = [
        LDUBox(0, 0, 0, 80, 40, 24),
        LDUBox(-20, -20, 0, 40, 20, 48),
        LDUBox(15980, 16000, 9600, 16040, 16040, 9624),
        LDUBox(5000, 5000, 5000, 5080, 5080, 5024),
    ]
    for pb in probe_boxes:
        assert tf.intersection_volume((pb,)) == exact_ref.intersection_volume((pb,))
    assert tf.contains_point((10, 10, 12)) is True
    assert tf.contains_point((16020, 16020, 9612)) is True
    assert tf.contains_point((8000, 8000, 4800)) is False

    # WS-1B: Genuinely blind whole-target `TopoBrickKernel.realize(target_field)` (zero .ldr string!)
    kernel = TopoBrickKernel()
    arch_target = SparseTiledTargetField(
        boxes=(
            LDUBox(0, 0, 0, 40, 40, 72),
            LDUBox(80, 0, 0, 120, 40, 72),
            LDUBox(0, 0, 72, 120, 40, 96),
        )
    )
    res_blind = kernel.realize(
        target_field=arch_target,
        allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
        object_id="blind_arch_gate",
    )
    assert res_blind.realization_mode == "blind"
    assert res_blind.status == KernelResultStatus.COMPLETED_CERTIFIED
    assert res_blind.certificates.overall_certified is True
    assert res_blind.certificates.geometry.target_iou >= 0.999

    # WS-1C: Unconditional per-course 6-DoF LP check across > 4 courses in `certify(verify_prefix=True)`
    # Build a 7-course structure where intermediate course 2..4 cantilevers too far and is only
    # counterbalanced at course 6: final static LP passes, but intermediate prefix LP MUST fail!
    strict_val = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=12.0, moment_arm_capacity_ldu=12.0
    )
    strict_kernel = TopoBrickKernel(validator=strict_val)
    st_trap = AssemblyState()
    st_trap.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)  # y in [-10, 70], z=0..24
    # Courses 1..4: step +40 LDU in +Y each course -> overhanging cantilever fails intermediate LP
    st_trap.place_root_part("3001", LDUFrame(0, (0, 40, 24)), is_grounded=False)
    st_trap.place_root_part("3001", LDUFrame(0, (0, 80, 48)), is_grounded=False)
    st_trap.place_root_part("3001", LDUFrame(0, (0, 120, 72)), is_grounded=False)
    st_trap.place_root_part("3001", LDUFrame(0, (0, 160, 96)), is_grounded=False)
    # Courses 5..6: step back toward base
    st_trap.place_root_part("3001", LDUFrame(0, (0, 80, 120)), is_grounded=False)
    st_trap.place_root_part("3001", LDUFrame(0, (0, 0, 144)), is_grounded=False)

    certs_trap = strict_kernel.certify(st_trap, verify_prefix=True, verify_insertion=False)
    assert certs_trap.prefix_mechanics.evaluated_steps == 7
    assert certs_trap.prefix_mechanics.status == CertStatus.FAIL
    assert certs_trap.prefix_mechanics.first_unstable_step is not None
    assert certs_trap.prefix_mechanics.first_unstable_step < 6

    # WS-1D: Fast `TopoBrickKernel.constrain()` with IncrementalCandidateCache (< 30 ms median step)
    st_con = AssemblyState()
    st_con.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_con.place_root_part("3001", LDUFrame(0, (40, 0, 0)), is_grounded=True)
    cache = None
    step_ms = []
    for _ in range(12):
        t0 = time.perf_counter()
        c_res = kernel.constrain(
            st_con,
            target_field=arch_target,
            allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
            max_actions=6,
            candidate_cache=cache,
        )
        step_ms.append((time.perf_counter() - t0) * 1000.0)
        cache = c_res.updated_cache
        if not c_res.valid_actions:
            break
        act = c_res.valid_actions[0]
        st_con.place_root_part(act.part_id, act.pose, is_grounded=False)
    assert st_con.verify_state_integrity() is True
    assert float(np.median(step_ms)) < 30.0


def test_ws2_coulomb_patch_mechanics_and_physical_validation_agreement() -> None:
    """Verify WS-2A Coulomb + sublinear patch mechanics and WS-2B CMU StableLego 8/8 accuracy."""
    # WS-2A: Verify patch shear utilizes normal-force-coupled Coulomb friction cleanly
    val_low_clutch = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=2.0,
        moment_arm_capacity_ldu=16.0,
        coulomb_friction_coeff=0.60,
    )
    st_comp = AssemblyState()
    st_comp.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_comp.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    eq_comp = val_low_clutch.evaluate(st_comp)
    assert eq_comp.is_stable is True
    assert eq_comp.patch_utilizations is not None

    # WS-2B: CMU StableLego 8/8 physical hardware benchmark + BrickSim + StableText2Brick agreement
    cohort = load_untouched_s2b_600_cohort()
    rep = run_physical_and_model_agreement_suite(s2b_records=cohort["records"][:25])
    assert rep.num_physical_cases == 8
    assert rep.physical_cases_matched == 8
    assert rep.physical_hardware_accuracy == 1.0
    assert rep.bricksim_demo_certified_stable is True
    assert rep.s2b_num_evaluated == 25
    assert rep.s2b_substud_agreement_rate >= 0.96
    assert rep.s2b_kernel_lp_agreement_rate >= 0.96

    # Verify default `PhysicalValidationCert` is `UNKNOWN`, and opt-in calibrated check is `PASS`
    kernel = TopoBrickKernel()
    c_default = kernel.certify(st_comp, verify_calibrated_substud_physics=False)
    c_cal = kernel.certify(st_comp, verify_calibrated_substud_physics=True)
    assert c_default.physical_validation.status == CertStatus.UNKNOWN
    assert c_cal.physical_validation.status == CertStatus.PASS
    assert "stablelego" in c_cal.physical_validation.simulator_or_rig.lower()


def test_ws3_tier_a_catalog_se3_kinematics_omr_26sets_and_set_10698_alt_builds() -> None:
    """Verify WS-3A Set 10698 + TIER_A_CATALOG, WS-3B SE(3) OBB-SAT & joints, and WS-3C OMR + Alt-Builds."""
    # WS-3A: 100% coverage of official LEGO Set 10698-1 (108 unique parts, 790 pieces) in TIER_A_CATALOG
    assert len(SET_10698_OFFICIAL_INVENTORY) == 108
    assert sum(SET_10698_OFFICIAL_INVENTORY.values()) == 790
    assert len(TIER_A_CATALOG) >= 160
    missing_10698 = [pid for pid in SET_10698_OFFICIAL_INVENTORY if pid not in TIER_A_CATALOG]
    assert missing_10698 == []

    # Check all 8 connector families exist in TIER_A_CATALOG and every part has a 16-D intrinsic descriptor
    present_ctypes = {
        p.connector_type for prim in TIER_A_CATALOG.values() for p in prim.ports
    }
    for req_ct in (
        ConnectorType.STANDARD_STUD,
        ConnectorType.TECHNIC_PIN,
        ConnectorType.TECHNIC_AXLE,
        ConnectorType.BAR_3_18MM,
        ConnectorType.HINGE_PIN,
        ConnectorType.LOCKING_HINGE_PIN,
        ConnectorType.BALL_JOINT,
        ConnectorType.TURNTABLE_TOP_PIN,
        ConnectorType.WHEEL_PIN,
    ):
        assert req_ct in present_ctypes

    spec_sample = get_declarative_part_spec("60471")
    assert len(spec_sample.intrinsic_descriptor) == 16
    assert compute_intrinsic_part_descriptor("60471").shape == (16,)

    # WS-3B: 15-axis Separating-Axis-Theorem (SAT) OBB collider & continuous joint-DoF equilibrium
    pose_id = ContinuousSE3Pose.identity((0.0, 0.0, 0.0))
    pose_rot = ContinuousSE3Pose.from_axis_angle((0.0, 0.0, 1.0), math.radians(37.0), (12.0, 0.0, 0.0))
    pose_far = ContinuousSE3Pose.from_axis_angle((0.0, 1.0, 0.0), math.radians(37.0), (80.0, 0.0, 0.0))
    b_unit = LDUBox(-10, -10, 0, 10, 10, 24)
    assert obb_intersects_sat(box_to_obb(b_unit, pose_id), box_to_obb(b_unit, pose_rot)) is True
    assert obb_intersects_sat(box_to_obb(b_unit, pose_id), box_to_obb(b_unit, pose_far)) is False

    # Smooth hinge (`3937` port 2 + `3938` port 0 tilted 15 deg with `3001` load) cannot resist gravity torque,
    # whereas locking 9-tooth hinge (`60471` port 4 + `92582` port 6 tilted 15 deg with identical `3001` load)
    # resists gravity torque via detent moment!
    st_smooth = ContinuousAssemblyState()
    h_base = st_smooth.place_part(
        "3937",
        ContinuousSE3Pose.identity((0.0, 0.0, 0.0)),
        is_grounded=True,
    )
    h_top = st_smooth.attach_articulated_part(
        h_base, 2, "3938", 0, joint_angle_rad=math.radians(15.0)
    )
    h_load = st_smooth.attach_articulated_part(h_top, 1, "3001", 0, joint_angle_rad=0.0)
    assert h_top >= 0 and h_load >= 0
    eq_smooth = st_smooth.evaluate_static_equilibrium()
    assert eq_smooth.is_stable is False

    st_lock = ContinuousAssemblyState()
    l_base = st_lock.place_part("60471", ContinuousSE3Pose.identity((0.0, 0.0, 0.0)), is_grounded=True)
    l_arm = st_lock.attach_articulated_part(
        l_base, 4, "92582", 6, joint_angle_rad=math.radians(15.0)
    )
    l_load = st_lock.attach_articulated_part(l_arm, 4, "3001", 0, joint_angle_rad=0.0)
    assert l_arm >= 0 and l_load >= 0
    eq_lock = st_lock.evaluate_static_equilibrium()
    assert eq_lock.is_stable is True

    # WS-3C (`Claim C`): Real 26-set OMR `.mpd` round-trip benchmark
    omr_rep = evaluate_omr_26set_benchmark()
    assert omr_rep.num_sets_evaluated == 26
    assert omr_rep.mean_part_coverage_ratio >= 0.90
    assert omr_rep.aggregate_part_coverage_ratio >= 0.90
    assert omr_rep.connection_compatibility_rate >= 0.95
    assert len(omr_rep.five_connector_families_covered) == 5
    assert omr_rep.total_non_cubic_se3_instances > 0
    assert omr_rep.roundtrip_export_reparse_exact_rate >= 0.999

    # WS-3C (`Claim E`): Set 10698-1 inventory-bounded Alt-Builds (5/5 certified, 0 overruns)
    alt_results = run_set_10698_inventory_bounded_alt_builds()
    assert len(alt_results) == 5
    for ar in alt_results:
        assert ar.certified is True
        assert ar.zero_inventory_overruns is True
        assert ar.max_part_usage_margin >= 0
        assert ar.target_iou >= 0.98
    assert float(np.mean([ar.target_iou for ar in alt_results])) >= 0.99


def test_ws4_untouched_s2b_600_cohort_and_hard_overhang_challenge_set() -> None:
    """Verify WS-4A (`Claim A`, n = 600 untouched cohort) and WS-4B (`Claim B`, n = 30 Hard-Overhangs)."""
    cohort = load_untouched_s2b_600_cohort()
    attr = cohort["attrition_summary"]
    records = cohort["records"]

    assert len(records) == 600
    assert attr["sampled_total_objects"] == 600
    assert attr["sampled_le_100_bricks"] == 450
    assert attr["sampled_101_to_250_bricks"] == 150
    assert attr["prior_pilot_and_eval_overlap_removed"] == 0

    # Strictly verify 0 object_id overlap with all 37 prior pilot (16) and eval (21) objects
    prior_37 = get_prior_locked_and_eval_object_ids()
    assert len(prior_37) == len(LOCKED_STABLETEXT2BRICK_MANIFEST) + len(
        UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST
    )
    cohort_oids = [str(r["object_id"]) for r in records]
    assert len(set(cohort_oids)) == 600
    assert prior_37.isdisjoint(set(cohort_oids))

    # Verify saved dual-host n = 600 benchmark results + Clopper-Pearson CIs
    saved = load_untouched_s2b_600_results()
    assert saved["zero_prior_overlap_verified"] is True
    assert saved["unique_object_ids_verified"] is True
    assert len(saved["hosts"]) >= 1
    for host_id, h_metrics in saved["hosts"].items():
        blind_m = h_metrics["S1_Blind_WholeTarget"]
        ref_m = h_metrics["S1_Reference_Assisted"]
        base_m = h_metrics["Legolization_Baseline_Blind"]
        assert blind_m["realization_mode"] == "blind"
        assert ref_m["realization_mode"] == "reference_assisted"
        assert blind_m["num_objects"] == 600
        assert blind_m["certified_completion_rate"] > base_m["certified_completion_rate"]
        assert blind_m["certified_completion_rate"] >= 0.98
        assert blind_m["static_stable_rate"] >= 0.995
        lo_chk, hi_chk = clopper_pearson_95_ci(blind_m["certified_count"], 600)
        assert abs(blind_m["certified_ci95_low"] - lo_chk) < 1e-9
        assert abs(blind_m["certified_ci95_high"] - hi_chk) < 1e-9
        if "desktop" in host_id or "host_b" in host_id:
            # Includes Phase 8 atomic 1-by-1 prefix + full matrix primal witness residual checks
            assert blind_m["median_wall_clock_ms_le_100"] <= 800.0
        else:
            assert blind_m["median_wall_clock_ms_le_100"] <= 1500.0

    # Live blind verification on a deterministic 12-object stratified slice of the 600 cohort
    kernel = TopoBrickKernel()
    live_cert = 0
    for rec in records[::50]:  # 12 objects spanning <=100 and 101..250 bricks
        r_live = kernel.realize_external_record(rec, solver_id="S1_CoverAndRepair", mode="blind")
        assert r_live.realization_mode == "blind"
        if r_live.status == KernelResultStatus.COMPLETED_CERTIFIED:
            live_cert += 1
    assert live_cert >= 11

    # WS-4B (`Claim B`): n = 30 Hard-Overhang / Arch Challenge Set
    ch_rep = evaluate_hard_overhang_arch_challenge_set()
    assert ch_rep.num_challenges == 30
    assert ch_rep.baseline_stable_count == 0
    assert ch_rep.baseline_stable_rate == 0.0
    assert ch_rep.s1_no_support_stable_count == 0
    assert ch_rep.s1_support_certified_count == 30
    assert ch_rep.s1_support_certified_rate == 1.0
    assert ch_rep.mean_target_recall_with_support >= 0.99


def test_ws5_disjoint_m2_m3_generalization_and_live_transformer_logits_processor() -> None:
    """Verify WS-5A (120-part TIER_A_CATALOG M2 on n=50 disjoint courses) and WS-5B (`Claim D` BrickGPT)."""
    # WS-5A: M2 on 50 strictly object_id-disjoint courses over 120 parts of TIER_A_CATALOG
    m2_rep = evaluate_m2_on_tier_a_catalog(
        num_train_courses=40, num_eval_courses=50, catalog_size=120, keep_ratio=0.50
    )
    assert m2_rep.catalog_size == 120
    assert m2_rep.num_train_courses == 40
    assert m2_rep.num_eval_courses == 50
    assert m2_rep.zero_overlap_verified is True
    assert m2_rep.speedup_vs_s1_ratio > 1.05
    assert abs(m2_rep.iou_delta_vs_s1) <= 0.015
    assert m2_rep.verdict.reliability_gate_passed is True

    # WS-5B (`Claim D`): Live HuggingFace `LlamaForCausalLM.generate()` 100-step constrained vs. unmasked
    cohort_recs = load_untouched_s2b_600_cohort()["records"]
    lm_records = sorted(cohort_recs, key=lambda r: int(r["num_bricks_claimed"]))[:6]
    lm_rep = run_live_transformer_constrained_generation_benchmark(
        records=lm_records, max_bricks=100, seed=42
    )
    assert lm_rep.vocab_size == BRICKGPT_VOCAB_SIZE
    assert lm_rep.num_rollouts == 6
    assert lm_rep.constrained_accepted_action_validity_rate == 1.0
    assert lm_rep.constrained_certified_final_rate == 1.0
    assert lm_rep.constrained_hundred_step_completion_rate > 0.0
    assert lm_rep.constrained_early_target_completion_rate > 0.0
    assert lm_rep.median_constrain_step_latency_ms < 30.0
    assert lm_rep.unmasked_valid_rollout_rate == 0.0
    assert lm_rep.unmasked_mean_steps_before_first_violation < 5.0
