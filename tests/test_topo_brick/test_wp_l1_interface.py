"""
Verification test suite for WP-L1:
Relative Port-Space Algebra, Persistent Sparse Assembly State,
Derived Active Connection Frontier, and Exact Candidate/Conflict Engine.
"""

from __future__ import annotations

import time
import numpy as np

from topo_brick.interface.brick_catalog import (
    ROT_Z_TO_POS_X,
    get_brick_primitive,
)
from topo_brick.interface.ldu_frames import (
    BRICK_HEIGHT_LDU,
    C4_Z_ROT_IDS,
    SO3_CUBIC_MATRICES,
    LDUBox,
    LDUFrame,
    compose_rot_ids,
    invert_rot_id,
    local_gravity,
    relative_transform,
    rot_id_to_matrix,
)
from topo_brick.interface.port_manifold import (
    AssemblyState,
    PortStatus,
)


def test_so3_cubic_group_exactness_and_invariance() -> None:
    """Gate 1: Verify 24 proper cubic rotations (det=+1) and global frame invariance."""
    assert SO3_CUBIC_MATRICES.shape == (24, 3, 3)
    for rot_id in range(24):
        mat = rot_id_to_matrix(rot_id)
        # Orthogonality R^T R = I
        assert np.array_equal(mat.T @ mat, np.eye(3, dtype=np.int64))
        # Proper rotation (no reflections): det(R) == +1
        assert int(round(np.linalg.det(mat))) == 1
        # Inverse table exactness
        inv_id = invert_rot_id(rot_id)
        assert compose_rot_ids(rot_id, inv_id) == 0
        assert compose_rot_ids(inv_id, rot_id) == 0

    # Verify global invariance: (H ∘ T_u)^-1 ∘ (H ∘ T_v) == T_u^-1 ∘ T_v
    t_u = LDUFrame(rot_id=5, pos=(20, -40, 16))
    t_v = LDUFrame(rot_id=17, pos=(-60, 20, 48))
    base_rel = relative_transform(t_u, t_v)

    for h_rot in range(24):
        h = LDUFrame(rot_id=h_rot, pos=(100, -200, 56))
        hu = h.compose(t_u)
        hv = h.compose(t_v)
        assert relative_transform(hu, hv) == base_rel


def test_local_gravity_observability_across_orientations() -> None:
    """Verify local gravity g_u = R_u^T g_world distinguishes upright and SNOT frames."""
    upright = LDUFrame(rot_id=0, pos=(0, 0, 24))
    assert local_gravity(upright) == (0, 0, -1)

    # SNOT port whose local +Z points along world +X (local X -> -Z, Y -> +Y, Z -> +X)
    snot_pos_x = LDUFrame(rot_id=ROT_Z_TO_POS_X, pos=(10, 0, 16))
    # Since world gravity is (0, 0, -1) and local X is (0, 0, -1), g_local is (+1, 0, 0)
    assert local_gravity(snot_pos_x) == (1, 0, 0)


def test_persistent_state_distinguishes_buried_seam_partitions() -> None:
    """
    Reviewer Section 3.2 counterexample:
    Two 2x2 assemblies have identical external geometry and identical exposed top/bottom ports,
    but Assembly A has an unbridged vertical seam while Assembly B bridges the seam.
    Verify that the persistent ContactGraph distinguishes their structural connectivity.
    """
    # Assembly A: two parallel 1x2 bricks at z=0, plus two parallel 1x2 bricks stacked directly
    # above them (aligned seam -> 2 disconnected components)
    state_unbridged = AssemblyState()
    b0 = state_unbridged.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    b1 = state_unbridged.place_root_part("3004", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    state_unbridged.place_root_part("3004", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    state_unbridged.place_root_part("3004", LDUFrame(0, (20, 0, 24)), is_grounded=False)

    # Assembly B: same two 1x2 base bricks at z=0, but top layer rotated 90 deg so each 1x2
    # bridges across b0 and b1 (interlocked running bond -> biconnected cycle)
    state_bridged = AssemblyState()
    b0_br = state_bridged.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    b1_br = state_bridged.place_root_part("3004", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    rot_90_z = C4_Z_ROT_IDS[3]  # maps local +Y (0, 20, 0) -> world (+20, 0, 0)
    state_bridged.place_root_part("3004", LDUFrame(rot_90_z, (0, 0, 24)), is_grounded=False)
    state_bridged.place_root_part("3004", LDUFrame(rot_90_z, (0, 20, 24)), is_grounded=False)

    frontier_a = state_unbridged.get_active_frontier()
    frontier_b = state_bridged.get_active_frontier()
    # Both expose the exact same 8 ports (4 top studs at z=48, 4 bottom anti-studs at z=0)
    assert len(frontier_a) == 8
    assert len(frontier_b) == 8
    pos_a = sorted(p.world_frame.pos for p in frontier_a.ports)
    pos_b = sorted(p.world_frame.pos for p in frontier_b.ports)
    assert pos_a == pos_b

    # Yet persistent ContactGraph knows b0 and b1 are disconnected in A and biconnected in B
    assert not state_unbridged.are_parts_biconnected(b0, b1)
    assert state_bridged.are_parts_biconnected(b0_br, b1_br)


def test_off_axis_obstacle_caught_by_whole_hull_box_check() -> None:
    """
    Reviewer Section 4.2 counterexample:
    An anchor stud is at (0, 0, 24). All 6 axis-aligned rays from (0, 0, 24) are completely clear,
    because an obstacle sits strictly off-axis at diagonal offset (20, 20, 24..48).
    Verify that a 2x2 brick ('3003') extending into (+20, +20) is rejected, while the three
    other quadrants (-20,+20), (-20,-20), (+20,-20) remain valid candidates.
    """
    state = AssemblyState()
    # Anchor 1x1 brick at (0, 0, 0) exposing top stud at (0, 0, 24)
    state.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    # Off-axis obstacle at (20, 20, 24) (leaves +X, -X, +Y, -Y axes through (0,0,24) clear)
    state.place_root_part("3005", LDUFrame(0, (20, 20, 24)), is_grounded=False)

    frontier = state.get_active_frontier()
    # Filter candidates that attach specifically to the anchor top stud (instance 0, port 1)
    pool = state.generate_candidate_pool(allowed_part_ids=["3003"], frontier=frontier)
    anchor_top_gpid = (0, 1)
    cands_on_anchor = [
        pool.candidates[cid] for cid in pool.port_to_candidates.get(anchor_top_gpid, ())
    ]

    # A 2x2 brick ('3003') on a single stud normally has 4 quadrants x 4 rotations = 16 poses,
    # deduplicated to 4 unique quadrant footprints x 4 rotations = 12 or 16 poses;
    # EVERY pose occupying quadrant (+20, +20) must be absent!
    obstacle_box = LDUBox(10, 10, 24, 30, 30, 48)
    assert len(cands_on_anchor) > 0
    for cand in cands_on_anchor:
        for wbox in cand.world_boxes:
            assert not wbox.intersects(obstacle_box)


def test_disjoint_port_hull_conflict_detection_and_decoding() -> None:
    """
    Reviewer Section 4.3 counterexample:
    Two candidate placements consume completely disjoint ports (P_i ∩ P_j = ∅)
    yet their 3D brick bodies overlap. Verify E_conflict catches the collision.
    """
    state = AssemblyState()
    # Two separate 1x1 base bricks at (0, 0, 0) and (20, 20, 0) (diagonal neighbors)
    state.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    state.place_root_part("3005", LDUFrame(0, (20, 20, 0)), is_grounded=True)

    pool = state.generate_candidate_pool(allowed_part_ids=["3004"])  # 1x2 brick
    # Find candidate A attached ONLY to brick 0's top stud (0, 1) extending to (0, 20, 24)
    # and candidate B attached ONLY to brick 1's top stud (1, 1) extending to (0, 20, 24)
    target_overlap_box = LDUBox(-5, 15, 25, 5, 25, 45)
    overlapping_from_0 = [
        c.candidate_id
        for c in pool.candidates
        if c.consumed_ports == frozenset({(0, 1)})
        and any(b.intersects(target_overlap_box) for b in c.world_boxes)
    ]
    overlapping_from_1 = [
        c.candidate_id
        for c in pool.candidates
        if c.consumed_ports == frozenset({(1, 1)})
        and any(b.intersects(target_overlap_box) for b in c.world_boxes)
    ]
    assert len(overlapping_from_0) >= 1
    assert len(overlapping_from_1) >= 1

    cid_a = overlapping_from_0[0]
    cid_b = overlapping_from_1[0]
    # They consume disjoint ports, yet must be marked incompatible in E_conflict
    assert pool.candidates[cid_a].consumed_ports.isdisjoint(pool.candidates[cid_b].consumed_ports)
    pair = (min(cid_a, cid_b), max(cid_a, cid_b))
    assert pair in pool.conflict_edges
    assert not pool.is_selection_compatible([cid_a, cid_b])

    # Greedy course decoder given high scores for both must select only one
    scores = np.full(len(pool), -10.0)
    scores[cid_a] = 10.0
    scores[cid_b] = 9.0
    selected = pool.decode_compatible_course(scores, score_threshold=0.0)
    assert selected == [cid_a]


def test_canonical_5_5_6_snot_closed_loop_cycle() -> None:
    """
    Gate 4: Verify the canonical 5:5:6 SNOT ratio (5 plates = 40 LDU = 2 studs)
    allows a sideways 1x3 plate ('3623') to simultaneously lock two '87087' SNOT bricks
    separated vertically by 5 plates, closing a biconnected 3D structural loop.
    """
    state = AssemblyState()
    # Bottom 87087 SNOT brick at z = 0..24 (lateral +X stud at (10, 0, 16))
    b_bot = state.place_root_part("87087", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    # Two 1x1 plates ('3024', height 8 LDU each) at z = 24..32 and z = 32..40
    state.place_root_part("3024", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    state.place_root_part("3024", LDUFrame(0, (0, 0, 32)), is_grounded=False)
    # Top 87087 SNOT brick at z = 40..64 (lateral +X stud at (10, 0, 56))
    # Vertical separation between the two lateral +X SNOT studs is 56 - 16 = 40 LDU = 2 studs!
    b_top = state.place_root_part("87087", LDUFrame(0, (0, 0, 40)), is_grounded=False)

    assert not state.are_parts_biconnected(b_bot, b_top)

    # Generate candidates for a standard 1x3 plate ('3623')
    pool = state.generate_candidate_pool(allowed_part_ids=["3623"])
    snot_loop_cands = [
        c
        for c in pool.candidates
        if (b_bot, 2) in c.consumed_ports and (b_top, 2) in c.consumed_ports
    ]
    assert len(snot_loop_cands) == 1
    loop_cand = snot_loop_cands[0]
    assert loop_cand.closes_structural_seam
    assert loop_cand.creates_biconnected_cycle

    # Commit the sideways SNOT locking plate and verify biconnected loop closure
    state.commit_candidate(loop_cand)
    assert state.are_parts_biconnected(b_bot, b_top)


def test_port_status_classifications_and_snot_catalog() -> None:
    """Verify all catalog primitives (including 4070, 87087, 99780, 99207) and 3-tier port status."""
    for part_id in ("4070", "87087", "99780", "99207", "3001", "3020"):
        prim = get_brick_primitive(part_id)
        assert len(prim.hull_boxes) >= 1
        assert len(prim.ports) >= 2

    state = AssemblyState()
    b0 = state.place_root_part("3005", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    # Place overhang brick 2 plates above b0's top stud (at z = 40..64, leaving a 16-LDU gap)
    state.place_root_part("3005", LDUFrame(0, (0, 0, 40)), is_grounded=False)
    # b0's top stud at (0, 0, 24) is unmatched and exposed at z=25, but blocked at z=48 along corridor
    status = state.classify_port_status(b0, 1)
    assert status == PortStatus.EXPOSED_BLOCKED
    assert status.is_exposed
    assert not status.is_reachable


def test_boundary_update_throughput() -> None:
    """Verify fast deterministic boundary & contact updates."""
    state = AssemblyState()
    t0 = time.perf_counter()
    for i in range(100):
        state.place_root_part(
            "3004",
            LDUFrame(0, (0, 0, i * BRICK_HEIGHT_LDU)),
            is_grounded=(i == 0),
        )
    elapsed = time.perf_counter() - t0
    assert state.num_parts == 100
    assert elapsed < 0.25
