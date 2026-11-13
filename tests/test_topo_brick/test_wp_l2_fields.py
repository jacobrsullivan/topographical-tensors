"""
Verification test suite for WP-L2:
Topographical Frontier Port Tensor (24-ch), Sparse Relative-Frame Edge Graph,
and Candidate Feature Tensor (18-ch).
"""

from __future__ import annotations

import numpy as np

from topo_brick.interface.ldu_frames import (
    C4_Z_ROT_IDS,
    LDUBox,
    LDUFrame,
)
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.representation.interface_fields import (
    CANDIDATE_FEATURE_CHANNELS,
    FRONTIER_PORT_CHANNELS,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)


def test_frontier_tensor_shapes_and_buried_seam_discrimination() -> None:
    """
    WP-L2 Gate 1: Verify tensor dimensions and confirm that Ch 15 (seam_bridge_demand)
    and Ch 16 (biconnected_ratio) distinguish two assemblies with identical exposed
    frontier ports but unbridged vs. bridged buried internal seams (Reviewer §3.2).
    """
    # Assembly A: unbridged vertical seam
    state_a = AssemblyState()
    state_a.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    state_a.place_root_part("3004", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    state_a.place_root_part("3004", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    state_a.place_root_part("3004", LDUFrame(0, (20, 0, 24)), is_grounded=False)

    # Assembly B: bridged running-bond seam
    state_b = AssemblyState()
    state_b.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    state_b.place_root_part("3004", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    rot_90 = C4_Z_ROT_IDS[3]
    state_b.place_root_part("3004", LDUFrame(rot_90, (0, 0, 24)), is_grounded=False)
    state_b.place_root_part("3004", LDUFrame(rot_90, (0, 20, 24)), is_grounded=False)

    batch_a = build_frontier_and_candidate_tensors(state_a)
    batch_b = build_frontier_and_candidate_tensors(state_b)

    assert batch_a.port_features.shape == (8, FRONTIER_PORT_CHANNELS)
    assert batch_b.port_features.shape == (8, FRONTIER_PORT_CHANNELS)
    assert batch_a.candidate_features.shape[1] == CANDIDATE_FEATURE_CHANNELS

    # Unbridged Assembly A has high seam-bridge demand (Ch 15 > 0) and low biconnected ratio (Ch 16 < 0.5)
    # Bridged Assembly B has zero unbraced seam demand (Ch 15 == 0) and 100% biconnected ratio (Ch 16 == 1.0)
    assert float(np.mean(batch_a.port_features[:, 15])) > 0.4
    assert float(np.mean(batch_b.port_features[:, 15])) == 0.0
    assert float(np.mean(batch_a.port_features[:, 16])) < 0.4
    assert np.allclose(batch_b.port_features[:, 16], 1.0)


def test_exact_quadrant_feasibility_channel_under_off_axis_obstacle() -> None:
    """
    WP-L2 Gate 2: Verify that Ch 10 (2x2/2x4 quadrant feasibility ratio) drops from 1.0 to 0.75
    when an off-axis diagonal obstacle blocks one of the 4 quadrants around an anchor stud.
    """
    state_clear = AssemblyState()
    state_clear.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    batch_clear = build_frontier_and_candidate_tensors(state_clear)
    # Port index 1 is the top stud at (0, 0, 24)
    assert np.isclose(batch_clear.port_features[1, 10], 1.0)

    state_blocked = AssemblyState()
    state_blocked.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    state_blocked.place_root_part("3005", LDUFrame(0, (20, 20, 24)), is_grounded=False)
    batch_blocked = build_frontier_and_candidate_tensors(state_blocked)
    assert np.isclose(batch_blocked.port_features[1, 10], 0.75)


def test_global_translation_and_yaw_invariance_of_frontier_tensors() -> None:
    """
    WP-L2 Gate 3: Verify that translating and rotating the assembly + target shape by any
    gravity-preserving transform H in Z^3_LDU ⋊ C_4 preserves all frontier port features,
    relative edge transforms, and candidate features.
    """
    state_0 = AssemblyState()
    state_0.place_root_part("87087", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    state_0.place_root_part("3004", LDUFrame(0, (0, 20, 0)), is_grounded=True)
    target_0 = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 10, 50, 48),))
    batch_0 = build_frontier_and_candidate_tensors(state_0, target_shape=target_0)

    # Apply global transform H: yaw rotation by 90 deg + translation (120, -80, 48)
    h_frame = LDUFrame(rot_id=C4_Z_ROT_IDS[1], pos=(120, -80, 48))
    state_h = AssemblyState()
    state_h.place_root_part("87087", h_frame.compose(LDUFrame(0, (0, 0, 0))), is_grounded=True)
    state_h.place_root_part("3004", h_frame.compose(LDUFrame(0, (0, 20, 0))), is_grounded=True)
    target_h = TargetShapeBoxes(boxes=(target_0.boxes[0].transform(h_frame),))
    batch_h = build_frontier_and_candidate_tensors(state_h, target_shape=target_h)

    assert np.allclose(batch_0.port_features, batch_h.port_features, atol=1e-6)
    assert np.array_equal(batch_0.edge_rel_rot_id, batch_h.edge_rel_rot_id)
    assert np.allclose(batch_0.edge_rel_pos_ldu, batch_h.edge_rel_pos_ldu, atol=1e-6)
    assert np.allclose(batch_0.edge_attr, batch_h.edge_attr, atol=1e-6)
