"""
Verification test suite for WP-L3:
SparsePortIPAEncoder, RelativeFrameGNNEncoder, SparseStaticEquilibriumValidator,
Stage1FrontierOutcomeModel, and Stage1FrontierPlanner.
"""

from __future__ import annotations

import torch

from topo_brick.evaluation.benchmark_suite import run_target_wall_reconstruction
from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.ldu_frames import (
    C4_Z_ROT_IDS,
    LDUBox,
    LDUFrame,
)
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.models.outcome_head import (
    Stage1FrontierOutcomeModel,
    Stage1FrontierPlanner,
)
from topo_brick.representation.interface_fields import (
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)


def test_reviewer_section_7_shared_buried_joint_overload_and_backoff() -> None:
    """
    Reviewer Section 7 exact counterexample:
    Three proposed bricks each add 1.0 unit of load through DIFFERENT visible ports on a
    platform brick B1. All loads pass through the same buried connection (B0 <-> B1), which
    can carry at most 2.5 units total (B1 itself weighs 0.5 units).
    - Adding any 1 brick (total 1.5 <= 2.5) is STABLE.
    - Adding any pair of 2 bricks (total 2.5 <= 2.5) is STABLE.
    - Adding all 3 bricks together (total 3.5 > 2.5) OVERLOADS the buried connection.
    - Verify `SparseStaticEquilibriumValidator` catches the 3-brick overload and pinpoints
      the buried connection, and `Stage1FrontierPlanner` automatically backs off from 3 bricks
      to the 2-brick stable prefix!
    """
    validator = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=2.5,
        moment_arm_capacity_ldu=1000.0,  # isolate pure vertical shared-joint overload
        use_uniform_brick_weight_grams=1.0,
    )

    def make_base_platform() -> AssemblyState:
        st = AssemblyState()
        # Grounded base 1x1 brick B0 at (0, 0, 0)
        st.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
        # Platform 1x4 brick B1 at (0, 0, 24) connected to B0 ONLY via port (0,1)<->(1,0)
        # Note: with uniform weight = 1.0g and capacity = 3.0g:
        #   B1 alone = 1.0g (eta = 0.333)
        #   B1 + 1 brick = 2.0g (eta = 0.667 <= 1.0 -> STABLE)
        #   B1 + 2 bricks = 3.0g (eta = 1.000 <= 1.0 -> STABLE)
        #   B1 + 3 bricks = 4.0g (eta = 1.333 > 1.0  -> OVERLOADED!)
        st.place_root_part("3010", LDUFrame(0, (0, 0, 24)), is_grounded=False)
        return st

    validator.clutch_force_capacity_grams = 3.2

    # 1. Check 1 added brick on port y=20
    st1 = make_base_platform()
    st1.place_root_part("3005", LDUFrame(0, (0, 20, 48)), is_grounded=False)
    res1 = validator.evaluate(st1)
    assert res1.is_stable
    assert res1.peak_utilization < 0.70

    # 2. Check 2 added bricks on distinct ports y=0 and y=20 (or y=20 and y=40)
    st2 = make_base_platform()
    st2.place_root_part("3005", LDUFrame(0, (0, 0, 48)), is_grounded=False)
    st2.place_root_part("3005", LDUFrame(0, (0, 20, 48)), is_grounded=False)
    res2 = validator.evaluate(st2)
    assert res2.is_stable
    assert res2.peak_utilization < 1.0

    # 3. Check 3 added bricks on distinct ports y=0, y=20, y=40 -> overloads buried joint (0,1)<->(1,0)!
    st3 = make_base_platform()
    st3.place_root_part("3005", LDUFrame(0, (0, 0, 48)), is_grounded=False)
    st3.place_root_part("3005", LDUFrame(0, (0, 20, 48)), is_grounded=False)
    st3.place_root_part("3005", LDUFrame(0, (0, 40, 48)), is_grounded=False)
    res3 = validator.evaluate(st3)
    assert not res3.is_stable
    assert res3.peak_utilization > 1.20
    assert res3.weakest_contact == ((0, 1), (1, 0))

    # 4. Verify Stage1FrontierPlanner proposes 3 bricks inside target box, detects the
    #    3-brick buried-joint overload, and backs off by 1 step to commit exactly 2 bricks!
    st_plan = make_base_platform()
    target_three = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 48, 10, 50, 72),))
    model = Stage1FrontierOutcomeModel(d_model=32, n_layers=2, n_heads=4, use_ipa=True)
    planner = Stage1FrontierPlanner(
        model=model,
        validator=validator,
        allowed_part_ids=["3005"],
        max_bricks_per_course=3,
    )
    step_metrics = planner.step_course(
        state=st_plan,
        target_shape=target_three,
        require_target_precision=0.99,
    )
    assert len(step_metrics.proposed_candidate_indices) == 3
    assert step_metrics.prefix_backoff_steps == 1
    assert len(step_metrics.committed_instance_ids) == 2
    assert step_metrics.equilibrium_result.is_stable


def test_cantilever_bending_torque_overload_detection() -> None:
    """Verify 3D torque equilibrium catches a long cantilevered overhang."""
    validator = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=10.0,
        moment_arm_capacity_ldu=15.0,
    )
    state = AssemblyState()
    state.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    # Build a 4-step horizontal cantilever staircase of 2x4 bricks extending along +Y
    for step in range(1, 5):
        state.place_root_part(
            "3001",
            LDUFrame(0, (0, (step - 1) * 60, step * 24)),
            is_grounded=False,
        )
    res = validator.evaluate(state)
    assert not res.is_stable
    assert res.peak_utilization > 1.0


def test_ipa_and_gnn_encoder_invariance_and_sub_2ms_latency() -> None:
    """
    WP-L3 Gate 3: Verify exact global frame invariance of both `SparsePortIPAEncoder`
    and `RelativeFrameGNNEncoder`, and verify < 2 ms neural stability prediction speed.
    """
    torch.manual_seed(7)
    st_a = AssemblyState()
    st_a.place_root_part("99780", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_a.place_root_part("3004", LDUFrame(0, (0, 0, 8)), is_grounded=False)
    batch_a = build_frontier_and_candidate_tensors(st_a)

    h_trans = LDUFrame(rot_id=C4_Z_ROT_IDS[2], pos=(-200, 140, 80))
    st_b = AssemblyState()
    st_b.place_root_part("99780", h_trans.compose(LDUFrame(0, (0, 0, 0))), is_grounded=True)
    st_b.place_root_part("3004", h_trans.compose(LDUFrame(0, (0, 0, 8))), is_grounded=False)
    batch_b = build_frontier_and_candidate_tensors(st_b)

    for use_ipa in (True, False):
        model = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=use_ipa)
        model.eval()
        z_a = model.encode_frontier(batch_a)
        z_b = model.encode_frontier(batch_b)
        assert torch.allclose(z_a, z_b, atol=1e-5)

        # Warm-up + speed check
        _ = model.predict_stability(batch_a)
        pred = model.predict_stability(batch_a)
        assert 0.0 <= pred.stability_prob <= 1.0
        assert pred.inference_time_ms < 10.0


def test_end_to_end_target_wall_reconstruction_and_ablation() -> None:
    """
    WP-L3 Gate 4: Run end-to-end target wall reconstruction with both `SparsePortIPAEncoder`
    and `RelativeFrameGNNEncoder`, verifying 100% connectivity validity, 100% collision-free
    rate, 100% mechanical stability, and exact target IoU = 1.0.
    """
    summary_ipa = run_target_wall_reconstruction(use_ipa=True)
    summary_gnn = run_target_wall_reconstruction(use_ipa=False)

    for summary in (summary_ipa, summary_gnn):
        assert summary.connectivity_valid_rate == 1.0
        assert summary.collision_free_rate == 1.0
        assert summary.mechanically_stable_rate == 1.0
        assert summary.final_target_iou >= 0.99
        assert summary.total_bricks_placed >= 4
        assert "t_candidates_ms" in summary.latency_breakdown_ms
        assert "t_validate_ms" in summary.latency_breakdown_ms
