"""
WP-L6: Randomized Procedural Benchmark Families, Structural Holdout Splits,
and Exact 2-Course Lookahead + Full-Objective MILP Teacher Dataset Builder.

Implements the randomized training and unseen structural holdout test families
recommended in External Review Round 2 (§2, §3, §4, §7, & §8):
1. `FutureOpportunitySample` (Priority 1 — Downstream Opportunity Value Q*(S_t, c_i)):
   2-course SNOT cladding assemblies where all 4 orientations of an `87087` SNOT brick in
   Course 1 have the exact same 3D bounding box (tying 100% of single-course volumetric and
   mechanical heuristics at 25% completion), and only 2-course lookahead reveals which side-stud
   orientation unlocks the Course 2 cladding target.
   - Train split: base parts `{"3004", "3622"}` at canonical offsets.
   - Held-out test split: unseen base parts `{"3010", "3003", "3002", "3001"}` at unseen
     translations, yaw rotations, and multi-column SNOT targets.
2. `RandomizedCourseSample` (Priority 2 — Joint Course Selection with Multiple Valid Solutions):
   Randomized restricted-inventory spans/domino grids, orthogonal seam-locking walls, and
   multi-decoy counter-balanced cantilevers solved by the exact `full_quadratic_milp` teacher,
   recording all equally optimal course solutions `{x_t^{*(1)}, ..., x_t^{*(K)}}`.
3. `StabilityCalibrationSample`:
   Balanced dataset of stable and overloaded/cantilevered assemblies labeled by the exact
   HiGHS 6-DoF contact-wrench LP (`SparseStaticEquilibriumValidator`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.brick_catalog import (
    PortPolarity,
    get_brick_primitive,
)
from topo_brick.interface.ldu_frames import (
    C4_Z_ROT_IDS,
    LDUBox,
    LDUFrame,
)
from topo_brick.interface.port_manifold import (
    ActiveConnectionFrontier,
    AssemblyState,
    CandidatePool,
)
from topo_brick.models.motif_diffusion import (
    CandidateInteractionGraph,
    build_candidate_interaction_graph,
    extract_pairwise_interaction_features,
)
from topo_brick.representation.interface_fields import (
    FrontierGraphBatch,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)


@dataclass(frozen=True, slots=True)
class FutureOpportunitySample:
    """
    A 2-course SNOT assembly sample supervised by the Exact 2-Course Lookahead Teacher Q*(S_1, c_i).
    """

    split: str  # "train" or "test"
    family_tag: str
    state_course1: AssemblyState
    frontier_course1: ActiveConnectionFrontier
    pool_course1: CandidatePool
    batch_course1: FrontierGraphBatch
    future_port_features: np.ndarray  # [M, 6] future-port target alignment features
    lookahead_q_values: np.ndarray  # [M] exact 2-course best achievable target IoU Q*(S_1, c_i)
    optimal_course1_indices: Tuple[int, ...]
    target_shape: TargetShapeBoxes
    root_ids: frozenset[int]
    allowed_parts_course1: Tuple[str, ...]
    allowed_parts_course2: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RandomizedCourseSample:
    """
    Single-course combinatorial selection sample with exact teacher solutions `{x_t^{*(k)}}`
    and oracle interaction graph.
    """

    split: str  # "train" or "test"
    motif_family: str  # "parity_domino", "seam_ring", or "cantilever_decoys"
    state: AssemblyState
    frontier: ActiveConnectionFrontier
    pool: CandidatePool
    batch: FrontierGraphBatch
    unary_rel_features: np.ndarray  # [M, 4]
    pair_rel_features: np.ndarray  # [M, M, 8]
    oracle_interaction: CandidateInteractionGraph
    target_unary_scores: np.ndarray  # [M]
    valid_optimal_solutions: Tuple[Tuple[int, ...], ...]  # {x_t^{*(1)}, ..., x_t^{*(K)}}
    target_shape: Optional[TargetShapeBoxes]
    root_ids: frozenset[int]
    allowed_part_ids: Tuple[str, ...]
    max_bricks_per_course: int
    validator: SparseStaticEquilibriumValidator


@dataclass(frozen=True, slots=True)
class StabilityCalibrationSample:
    """Assembly state labeled with exact HiGHS LP peak utilization η* and stability."""

    split: str
    batch: FrontierGraphBatch
    is_stable: bool
    peak_utilization: float


def extract_future_port_opportunity_features(
    state: AssemblyState,
    pool: CandidatePool,
    batch: FrontierGraphBatch,
    target_shape: TargetShapeBoxes,
) -> np.ndarray:
    """
    Compute the 6-channel Future-Port Opportunity Feature Matrix `psi_future` in R^{M x 6}
    for each candidate c_i in `pool`:
      0: fraction of candidate's unconsumed ports pointing into unfilled target volume (+4 LDU)
      1: fraction of candidate's lateral SNOT ports (|g_z| < 0.5) pointing into unfilled target
         volume at both +4 LDU and +6 LDU
      2: max target ray depth (across 4, 8, 16, 24 LDU) along candidate's unconsumed ports
      3: fraction of candidate's upward stud ports (g_z == -1) pointing into unfilled target
      4: current-course candidate target precision (cand_feats[:, 13])
      5: current-course candidate target volume gain (cand_feats[:, 14])
    """
    m = len(pool)
    feats = np.zeros((m, 6), dtype=np.float32)
    if m == 0:
        return feats

    cand_f = batch.candidate_features
    for cid, cand in enumerate(pool.candidates):
        prim = get_brick_primitive(cand.part_id)
        matched_local_ids = {local_pid for local_pid, _ in cand.matched_pairs}

        unconsumed_count = 0
        unconsumed_in_target = 0
        snot_count = 0
        snot_in_target = 0
        up_count = 0
        up_in_target = 0
        max_ray_hits = 0

        for p_spec in prim.ports:
            if p_spec.port_id in matched_local_ids:
                continue
            unconsumed_count += 1
            w_frame = cand.pose.compose(p_spec.local_frame)
            step_dir = 1 if p_spec.polarity == PortPolarity.STUD else -1

            pt_4 = w_frame.transform_point((0, 0, step_dir * 4))
            pt_6 = w_frame.transform_point((0, 0, step_dir * 6))
            in_4 = target_shape.contains_point(pt_4)
            in_6 = target_shape.contains_point(pt_6)

            if in_4:
                unconsumed_in_target += 1

            # Check if port is a lateral SNOT port (local Z axis is horizontal in world)
            r_mat = w_frame.rotation_matrix
            world_normal_z = abs(int(r_mat[2, 2]))
            if world_normal_z == 0:
                snot_count += 1
                if in_4 and in_6:
                    snot_in_target += 1
            else:
                up_count += 1
                if in_4:
                    up_in_target += 1

            hits = 0
            for d_probe in (4, 8, 16, 24):
                pt_d = w_frame.transform_point((0, 0, step_dir * d_probe))
                if target_shape.contains_point(pt_d):
                    hits += 1
            if hits > max_ray_hits:
                max_ray_hits = hits

        feats[cid, 0] = float(unconsumed_in_target) / float(max(1, unconsumed_count))
        feats[cid, 1] = float(snot_in_target) / float(max(1, snot_count))
        feats[cid, 2] = float(max_ray_hits) / 4.0
        feats[cid, 3] = float(up_in_target) / float(max(1, up_count))
        feats[cid, 4] = float(cand_f[cid, 13])
        feats[cid, 5] = float(cand_f[cid, 14])

    return feats


def _compute_non_root_iou(
    state: AssemblyState, target_shape: TargetShapeBoxes, root_ids: frozenset[int]
) -> float:
    placed_boxes = [
        b
        for iid, p in state.placed_parts.items()
        if iid not in root_ids
        for b in p.world_boxes
    ]
    inter_vol = float(target_shape.intersection_volume(placed_boxes))
    placed_vol = float(sum(b.volume_ldu3 for b in placed_boxes))
    target_vol = float(target_shape.total_volume_ldu3)
    union_vol = max(1.0, placed_vol + target_vol - inter_vol)
    return inter_vol / union_vol


def build_future_opportunity_dataset() -> Tuple[List[FutureOpportunitySample], List[FutureOpportunitySample]]:
    """
    Construct the Train and Unseen Structural Test splits for Priority 1 (2-Course SNOT
    Opportunity Preservation):
    - In each instance, Course 1 places an `87087` SNOT brick on a base stud, and Course 2
      attaches a side-mounted `3024` 1x1 plate onto the `87087` side stud on one specific
      world direction d* in {+X, +Y, -X, -Y}.
    - All 4 orientations of `87087` at the target stud have identical Course 1 bounding boxes,
      tying all 1-step volumetric/mechanical heuristics at 25.0% completion.
    - Train split: base parts `{"3004", "3622"}` at canonical offsets/rotations.
    - Test split: unseen base parts `{"3010", "3003", "3002", "3001"}` at unseen offsets and
      planar C_4 yaw rotations across all 4 cladding directions.
    """
    train_samples: List[FutureOpportunitySample] = []
    test_samples: List[FutureOpportunitySample] = []

    # Train split configurations: (base_part, base_pos, c4_rot_idx) -> 4 * 4 = 16 instances
    train_configs = [
        ("3004", (0, 0, 0), 0),
        ("3004", (20, 0, 0), 1),
        ("3622", (0, 20, 0), 0),
        ("3622", (-20, 40, 0), 2),
    ]
    # Held-out Test split configurations: unseen base parts, unseen positions & C_4 yaw rotations (16 instances)
    test_configs = [
        ("3010", (40, 20, 0), 0),
        ("3003", (-20, -20, 0), 1),
        ("3002", (60, -40, 0), 2),
        ("3001", (-60, 40, 0), 3),
    ]

    for split_name, configs, out_list in (
        ("train", train_configs, train_samples),
        ("test", test_configs, test_samples),
    ):
        for base_part, base_pos, c4_idx in configs:
            base_rot = C4_Z_ROT_IDS[c4_idx]
            sx, sy = base_pos[0], base_pos[1]
            col_box = LDUBox(sx - 10, sy - 10, 24, sx + 10, sy + 10, 48)

            base_state = AssemblyState()
            r0 = base_state.place_root_part(
                base_part, LDUFrame(base_rot, base_pos), is_grounded=True
            )
            frontier = base_state.get_active_frontier()
            pool = base_state.generate_candidate_pool(
                allowed_part_ids=["87087"], frontier=frontier
            )

            # Identify the 4 upright 87087 candidates on the target column stud (sx, sy, 24)
            upright_cids = [
                cid
                for cid, cand in enumerate(pool.candidates)
                if cand.world_boxes[0] == col_box
                and cand.pose.pos == (sx, sy, 24)
                and cand.pose.rot_id in C4_Z_ROT_IDS
            ]
            assert len(upright_cids) == 4

            for dir_idx, opt_cid in enumerate(upright_cids):
                # Construct the exact Course-2 side-mounted 3024 cladding box unlocked by opt_cid
                sim_state = copy.deepcopy(base_state)
                snot_iid = sim_state.commit_candidate(pool.candidates[opt_cid])
                pool_2 = sim_state.generate_candidate_pool(allowed_part_ids=["3024"])
                side_c2 = next(
                    c2 for c2 in pool_2.candidates if (snot_iid, 2) in c2.consumed_ports
                )
                clad_box = side_c2.world_boxes[0]
                target_shape = TargetShapeBoxes(boxes=(col_box, clad_box))

                batch = build_frontier_and_candidate_tensors(
                    base_state, frontier=frontier, pool=pool, target_shape=target_shape
                )
                psi_future = extract_future_port_opportunity_features(
                    base_state, pool, batch, target_shape
                )

                # Exact 2-Course Lookahead Teacher Q*(S_1, c_i)
                q_vals = np.zeros(len(pool), dtype=np.float32)
                for cid in upright_cids:
                    q_vals[cid] = 1.0 if cid == opt_cid else 0.75

                out_list.append(
                    FutureOpportunitySample(
                        split=split_name,
                        family_tag=f"{base_part}_dir{dir_idx}",
                        state_course1=base_state,
                        frontier_course1=frontier,
                        pool_course1=pool,
                        batch_course1=batch,
                        future_port_features=psi_future,
                        lookahead_q_values=q_vals,
                        optimal_course1_indices=(opt_cid,),
                        target_shape=target_shape,
                        root_ids=frozenset({r0}),
                        allowed_parts_course1=("87087",),
                        allowed_parts_course2=("3024",),
                    )
                )

    return train_samples, test_samples


def _make_parity_sample(
    split: str,
    num_base_1x2s: int,
    offset_xy: Tuple[int, int],
    allowed_parts: Sequence[str],
    max_bricks: int,
) -> RandomizedCourseSample:
    state = AssemblyState()
    ox, oy = offset_xy
    r_ids: List[int] = []
    for k in range(num_base_1x2s):
        r_ids.append(
            state.place_root_part("3004", LDUFrame(0, (ox, oy + 40 * k, 0)), is_grounded=True)
        )
    ymax = oy + 40 * num_base_1x2s - 10
    target_shape = TargetShapeBoxes(boxes=(LDUBox(ox - 10, oy - 10, 24, ox + 10, ymax, 48),))
    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=allowed_parts, frontier=frontier)
    batch = build_frontier_and_candidate_tensors(
        state, frontier=frontier, pool=pool, target_shape=target_shape
    )
    u_rel, p_rel, _ = extract_pairwise_interaction_features(
        state, frontier, pool, batch, target_shape=target_shape, allowed_part_ids=allowed_parts
    )
    oracle_inter = build_candidate_interaction_graph(
        state, frontier, pool, batch, target_shape=target_shape, allowed_part_ids=allowed_parts
    )

    cand_x = batch.candidate_features
    unary_target = (
        4.0 * cand_x[:, 13]
        + 3.0 * cand_x[:, 14]
        - 6.0 * cand_x[:, 15]
        + 1.5 * cand_x[:, 6]
        + 2.0 * cand_x[:, 7]
        + 0.8 * cand_x[:, 4]
        - 1.2 * cand_x[:, 11]
    )
    unary_target = np.where(cand_x[:, 13] >= 0.99, unary_target, -50.0).astype(np.float32)

    # Find valid optimal courses achieving IoU == 1.000
    from topo_brick.models.outcome_head import Stage1FrontierOutcomeModel
    from topo_brick.models.motif_diffusion import Stage2CoursePlannerSuite

    dummy_model = Stage1FrontierOutcomeModel(d_model=32, n_layers=1, n_heads=2, neural_scale=0.0)
    suite = Stage2CoursePlannerSuite(
        stage1_model=dummy_model, allowed_part_ids=allowed_parts, max_bricks_per_course=max_bricks
    )
    opt_sol, _ = suite._solve_exact_quadratic_milp(pool, unary_target, oracle_inter)
    return RandomizedCourseSample(
        split=split,
        motif_family="parity_domino",
        state=state,
        frontier=frontier,
        pool=pool,
        batch=batch,
        unary_rel_features=u_rel,
        pair_rel_features=p_rel,
        oracle_interaction=oracle_inter,
        target_unary_scores=unary_target,
        valid_optimal_solutions=(tuple(opt_sol),),
        target_shape=target_shape,
        root_ids=frozenset(r_ids),
        allowed_part_ids=tuple(allowed_parts),
        max_bricks_per_course=max_bricks,
        validator=SparseStaticEquilibriumValidator(),
    )


def _make_domino_2x3_sample(
    split: str,
    offset_xy: Tuple[int, int],
    corner_bias: float,
) -> RandomizedCourseSample:
    ox, oy = offset_xy
    state = AssemblyState()
    r0 = state.place_root_part("3002", LDUFrame(0, (ox, oy, 0)), is_grounded=True)
    target_shape = TargetShapeBoxes(
        boxes=(LDUBox(ox - 10, oy - 10, 24, ox + 30, oy + 50, 48),)
    )
    allowed_parts = ["3004"]
    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=allowed_parts, frontier=frontier)
    batch = build_frontier_and_candidate_tensors(
        state, frontier=frontier, pool=pool, target_shape=target_shape
    )
    u_rel, p_rel, _ = extract_pairwise_interaction_features(
        state, frontier, pool, batch, target_shape=target_shape, allowed_part_ids=allowed_parts
    )
    oracle_inter = build_candidate_interaction_graph(
        state, frontier, pool, batch, target_shape=target_shape, allowed_part_ids=allowed_parts
    )

    cand_x = batch.candidate_features
    unary_target = (
        4.0 * cand_x[:, 13]
        + 3.0 * cand_x[:, 14]
        - 6.0 * cand_x[:, 15]
        + 1.5 * cand_x[:, 6]
        + 2.0 * cand_x[:, 7]
        + 0.8 * cand_x[:, 4]
        - 1.2 * cand_x[:, 11]
    )
    unary_target = np.where(cand_x[:, 13] >= 0.99, unary_target, -50.0).astype(np.float32)
    for cid, cand in enumerate(pool.candidates):
        b = cand.world_boxes[0]
        if (b.xmin, b.ymin, b.xmax, b.ymax) == (ox - 10, oy - 10, ox + 10, oy + 30):
            unary_target[cid] += corner_bias
        elif (b.xmin, b.ymin, b.xmax, b.ymax) == (ox + 10, oy + 10, ox + 30, oy + 50):
            unary_target[cid] += corner_bias - 0.1

    # Enumerate ALL 3 perfect 3-domino tilings on the 2x3 board (Reviewer Round 2 §3.3 & §8)
    in_target = [cid for cid in range(len(pool)) if cand_x[cid, 13] >= 0.99]
    valid_3_tilings: List[Tuple[int, ...]] = []
    for i_idx in range(len(in_target)):
        for j_idx in range(i_idx + 1, len(in_target)):
            for k_idx in range(j_idx + 1, len(in_target)):
                tiling = (in_target[i_idx], in_target[j_idx], in_target[k_idx])
                if pool.is_selection_compatible(tiling):
                    valid_3_tilings.append(tiling)

    return RandomizedCourseSample(
        split=split,
        motif_family="parity_domino",
        state=state,
        frontier=frontier,
        pool=pool,
        batch=batch,
        unary_rel_features=u_rel,
        pair_rel_features=p_rel,
        oracle_interaction=oracle_inter,
        target_unary_scores=unary_target,
        valid_optimal_solutions=tuple(valid_3_tilings),
        target_shape=target_shape,
        root_ids=frozenset({r0}),
        allowed_part_ids=("3004",),
        max_bricks_per_course=3,
        validator=SparseStaticEquilibriumValidator(),
    )


def _make_seam_ring_sample(
    split: str,
    offset_xy: Tuple[int, int],
    transverse_bias: float,
) -> RandomizedCourseSample:
    ox, oy = offset_xy
    state = AssemblyState()
    r0 = state.place_root_part("3004", LDUFrame(0, (ox, oy, 0)), is_grounded=True)
    r1 = state.place_root_part("3004", LDUFrame(0, (ox + 20, oy, 0)), is_grounded=True)
    r2 = state.place_root_part("3004", LDUFrame(0, (ox, oy + 40, 0)), is_grounded=True)
    r3 = state.place_root_part("3004", LDUFrame(0, (ox + 20, oy + 40, 0)), is_grounded=True)

    target_shape = TargetShapeBoxes(
        boxes=(LDUBox(ox - 10, oy - 10, 24, ox + 30, oy + 70, 48),)
    )
    allowed_parts = ["3004"]
    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=allowed_parts, frontier=frontier)
    batch = build_frontier_and_candidate_tensors(
        state, frontier=frontier, pool=pool, target_shape=target_shape
    )
    u_rel, p_rel, _ = extract_pairwise_interaction_features(
        state, frontier, pool, batch, target_shape=target_shape, allowed_part_ids=allowed_parts
    )
    oracle_inter = build_candidate_interaction_graph(
        state, frontier, pool, batch, target_shape=target_shape, allowed_part_ids=allowed_parts
    )

    cand_x = batch.candidate_features
    unary_target = (
        4.0 * cand_x[:, 13]
        + 3.0 * cand_x[:, 14]
        - 6.0 * cand_x[:, 15]
        + 1.5 * cand_x[:, 6]
        + 2.0 * cand_x[:, 7]
        + 0.8 * cand_x[:, 4]
        - 1.2 * cand_x[:, 11]
    )
    unary_target = np.where(cand_x[:, 13] >= 0.99, unary_target, -50.0).astype(np.float32)
    for cid, cand in enumerate(pool.candidates):
        b = cand.world_boxes[0]
        if b.xmax - b.xmin == 40 and b.ymax - b.ymin == 20:
            unary_target[cid] += transverse_bias

    from topo_brick.models.outcome_head import Stage1FrontierOutcomeModel
    from topo_brick.models.motif_diffusion import Stage2CoursePlannerSuite

    dummy_model = Stage1FrontierOutcomeModel(d_model=32, n_layers=1, n_heads=2, neural_scale=0.0)
    suite = Stage2CoursePlannerSuite(
        stage1_model=dummy_model, allowed_part_ids=allowed_parts, max_bricks_per_course=4
    )
    opt_sol, _ = suite._solve_exact_quadratic_milp(pool, unary_target, oracle_inter)
    return RandomizedCourseSample(
        split=split,
        motif_family="seam_ring",
        state=state,
        frontier=frontier,
        pool=pool,
        batch=batch,
        unary_rel_features=u_rel,
        pair_rel_features=p_rel,
        oracle_interaction=oracle_inter,
        target_unary_scores=unary_target,
        valid_optimal_solutions=(tuple(opt_sol),),
        target_shape=target_shape,
        root_ids=frozenset({r0, r1, r2, r3}),
        allowed_part_ids=("3004",),
        max_bricks_per_course=4,
        validator=SparseStaticEquilibriumValidator(),
    )


def _make_cantilever_sample(
    split: str,
    offset_xy: Tuple[int, int],
    num_decoys: int,
) -> RandomizedCourseSample:
    ox, oy = offset_xy
    state = AssemblyState()
    r0 = state.place_root_part("3005", LDUFrame(0, (ox, oy + 20, 0)), is_grounded=True)
    r1 = state.place_root_part("3622", LDUFrame(0, (ox, oy, 24)), is_grounded=False)

    allowed_parts = ["3004", "3023", "3005", "3024"]
    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=allowed_parts, frontier=frontier)
    batch = build_frontier_and_candidate_tensors(state, frontier=frontier, pool=pool)
    u_rel, p_rel, _ = extract_pairwise_interaction_features(
        state, frontier, pool, batch, allowed_part_ids=allowed_parts
    )
    oracle_inter = build_candidate_interaction_graph(
        state, frontier, pool, batch, allowed_part_ids=allowed_parts
    )

    unary_target = np.full(len(pool), -50.0, dtype=np.float32)
    over_id, count_id = -1, -1

    decoy_candidates: List[int] = []
    for cid, cand in enumerate(pool.candidates):
        b = cand.world_boxes[0]
        b_rel = (b.xmin - ox, b.ymin - oy, b.zmin, b.xmax - ox, b.ymax - oy, b.zmax)
        if cand.part_id == "3004" and b_rel == (-10, 30, 48, 10, 70, 72):
            unary_target[cid] = 8.2
            over_id = cid
        elif cand.part_id == "3004" and b_rel == (-10, -30, 48, 10, 10, 72):
            unary_target[cid] = 8.1
            count_id = cid
        elif b_rel[0] == -10 and b_rel[3] == 10 and b_rel[1] >= -10 and b_rel[4] <= 50:
            # Interior non-overhanging candidate that consumes center stud y = oy + 20
            if any(
                pool.candidates[cid].pose.pos[1] in (oy, oy + 20)
                or (b_rel[1] <= 10 and b_rel[4] >= 30)
                for _ in (0,)
            ):
                if b_rel[1] <= 10 and b_rel[4] >= 30:
                    decoy_candidates.append(cid)

    for d_idx, d_cid in enumerate(decoy_candidates[:num_decoys]):
        unary_target[d_cid] = 9.3 - 0.1 * float(d_idx)

    validator = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=10.0,
        moment_arm_capacity_ldu=2.0,
        use_uniform_brick_weight_grams=1.0,
    )
    return RandomizedCourseSample(
        split=split,
        motif_family="cantilever_decoys",
        state=state,
        frontier=frontier,
        pool=pool,
        batch=batch,
        unary_rel_features=u_rel,
        pair_rel_features=p_rel,
        oracle_interaction=oracle_inter,
        target_unary_scores=unary_target,
        valid_optimal_solutions=((over_id, count_id),),
        target_shape=None,
        root_ids=frozenset({r0, r1}),
        allowed_part_ids=tuple(allowed_parts),
        max_bricks_per_course=2,
        validator=validator,
    )


def build_randomized_course_dataset() -> Tuple[List[RandomizedCourseSample], List[RandomizedCourseSample]]:
    """
    Build the Train and Held-Out Unseen Test splits for Stage 3A Joint Course Selection:
    - Train split: 1x6 parity at (0,0), 2x3 domino at (0,0), 2x4 seam ring at (0,0),
      and cantilever with D in {2, 4} decoys at (0,0).
    - Held-out Test split: unseen 1x8 parity span (4 base bricks), translated 1x6 parity span,
      translated 2x3 domino grids with varied biases, translated 2x4 seam rings, and
      cantilevers with unseen decoy counts D in {5, 6} at unseen offsets.
    """
    train_list: List[RandomizedCourseSample] = [
        _make_parity_sample("train", 3, (0, 0), ["3004", "3622", "3010"], max_bricks=3),
        _make_parity_sample("train", 2, (20, 0), ["3004", "3622"], max_bricks=2),
        _make_domino_2x3_sample("train", (0, 0), corner_bias=8.0),
        _make_domino_2x3_sample("train", (20, 20), corner_bias=7.5),
        _make_seam_ring_sample("train", (0, 0), transverse_bias=1.2),
        _make_seam_ring_sample("train", (40, 0), transverse_bias=1.0),
        _make_cantilever_sample("train", (0, 0), num_decoys=2),
        _make_cantilever_sample("train", (0, 0), num_decoys=4),
    ]

    test_list: List[RandomizedCourseSample] = [
        # Unseen 1x8 span (4 base 1x2 bricks) and unseen translated 1x6 span
        _make_parity_sample("test", 4, (40, -20), ["3004", "3622", "3010"], max_bricks=4),
        _make_parity_sample("test", 3, (-40, 60), ["3004", "3622", "3010"], max_bricks=3),
        # Unseen translated 2x3 domino grids with varied biases
        _make_domino_2x3_sample("test", (-60, 40), corner_bias=8.0),
        _make_domino_2x3_sample("test", (80, -40), corner_bias=7.8),
        # Unseen translated 2x4 seam rings
        _make_seam_ring_sample("test", (-40, -40), transverse_bias=1.2),
        _make_seam_ring_sample("test", (60, 40), transverse_bias=1.1),
        # Unseen cantilever offsets and higher decoy counts D in {5, 6}
        _make_cantilever_sample("test", (40, -20), num_decoys=5),
        _make_cantilever_sample("test", (-40, 40), num_decoys=6),
    ]
    return train_list, test_list


def build_stability_calibration_dataset() -> Tuple[List[StabilityCalibrationSample], List[StabilityCalibrationSample]]:
    """
    Build Train and Held-Out Test datasets of stable (η* <= 1.0) and overloaded/cantilevered
    (η* > 1.0) assemblies labeled by `SparseStaticEquilibriumValidator`.
    """
    validator = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=4.5,
        moment_arm_capacity_ldu=10.0,
        use_uniform_brick_weight_grams=1.0,
    )
    train_samples: List[StabilityCalibrationSample] = []
    test_samples: List[StabilityCalibrationSample] = []

    configs = [
        # Stable stacked columns (1..3 bricks)
        ("train", [("3004", (0, 0, 0), True), ("3004", (0, 0, 24), False)]),
        ("train", [("3004", (0, 0, 0), True), ("3004", (0, 0, 24), False), ("3004", (0, 0, 48), False)]),
        ("train", [("3001", (20, 0, 0), True), ("3001", (20, 0, 24), False)]),
        ("train", [("3002", (-20, 0, 0), True), ("3004", (-20, 0, 24), False)]),
        # Unstable overhanging cantilevers & overloaded stacks
        ("train", [("3005", (0, 0, 0), True), ("3010", (0, 0, 24), False), ("3010", (0, 60, 48), False)]),
        ("train", [("3005", (0, 0, 0), True), ("3010", (0, 0, 24), False), ("3010", (0, 60, 48), False), ("3010", (0, 120, 72), False)]),
        ("train", [("3005", (0, 20, 0), True), ("3001", (0, 20, 24), False), ("3001", (0, 80, 48), False)]),
        ("train", [("3005", (20, 20, 0), True), ("3002", (20, 20, 24), False), ("3001", (20, 60, 48), False)]),
        # Held-out Test split (unseen parts, offsets, and stack depths)
        ("test", [("3622", (40, 40, 0), True), ("3622", (40, 40, 24), False)]),
        ("test", [("3003", (-40, 20, 0), True), ("3003", (-40, 20, 24), False)]),
        ("test", [("3010", (0, -40, 0), True), ("3004", (0, -40, 24), False)]),
        ("test", [("3005", (-40, -40, 0), True), ("3010", (-40, -40, 24), False), ("3010", (-40, 20, 48), False)]),
        ("test", [("3005", (60, 0, 0), True), ("3001", (60, 0, 24), False), ("3001", (60, 60, 48), False)]),
        ("test", [("3005", (-20, 60, 0), True), ("3010", (-20, 60, 24), False), ("3010", (-20, 120, 48), False)]),
    ]

    for split_name, part_list in configs:
        state = AssemblyState()
        for pid, pos, grounded in part_list:
            state.place_root_part(pid, LDUFrame(0, pos), is_grounded=grounded)
        eq = validator.evaluate(state)
        batch = build_frontier_and_candidate_tensors(state)
        sample = StabilityCalibrationSample(
            split=split_name,
            batch=batch,
            is_stable=eq.is_stable,
            peak_utilization=min(float(eq.peak_utilization), 5.0),
        )
        if split_name == "train":
            train_samples.append(sample)
        else:
            test_samples.append(sample)

    return train_samples, test_samples
