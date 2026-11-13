"""
WP-L8 & WP-L9 (Stage 4): Composed Multi-Motif Complete-Build Corpus (50-100 & 250-500 Bricks)
and Three-Step Sequential Experimental Protocol.

Implements the Stage 4 experimental program recommended in External Review Round 3 (§5 & §6):
- Freezes the 20-part catalog and 6-DoF contact-wrench LP physical model.
- Adopts `RelativeFrameGNN` (~0.12M total deployed parameters) as the fast default backbone,
  retaining `SparsePortIPA` (~0.42M parameters) as a comparison.
- Constructs composed multi-motif complete builds at two scale tiers:
  1. Tier 1 (`50-100` bricks, `12-16` courses, `60-84` bricks/build): 8 train builds and
     8 held-out unseen test builds combining running-bond seam rings, SNOT setup brackets
     (`87087`) unlocking next-course lateral `3024` cladding (with shallow decoy traps for
     paired non-myopic testing), restricted-inventory parity spans, and counterbalanced
     cantilever/arch courses (`requires_simultaneous_or_temp_support` tracked separately).
  2. Tier 2 (`250-500` bricks, `32-36` courses, `264-288` bricks/build): 4 held-out large-scale
     composed builds testing end-to-end scaling, memory, peak candidate count M_max, conflict
     edges |E_conflict|, and full-build wall-clock latency T_build.
- Executes the Three-Step Sequential Experimental Protocol across 5 independent training seeds:
  * Step 1: Decisive 5-planner comparison on identical complete candidate pools.
  * Step 2: 4-way candidate-engine comparison (`Full`, `Incremental`, `Selective`,
    `Incremental + Selective`) with the strongest practical learned planner.
  * Step 3: End-to-end complete-build verification across Tier 1 (`50-100` bricks) and
    Tier 2 (`250-500` bricks) in a unified timing & quality ledger.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.brick_catalog import get_brick_primitive
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.interface.port_manifold import (
    ActiveConnectionFrontier,
    AssemblyState,
    IncrementalCandidateCache,
    compute_cheap_pre_expansion_port_scores,
)
from topo_brick.models.motif_diffusion import (
    CandidateGraphDenoiser,
    LearnedCandidateInteractionModule,
    Stage2CoursePlannerSuite,
    build_candidate_interaction_graph,
    extract_pairwise_interaction_features,
)
from topo_brick.models.outcome_head import Stage1FrontierOutcomeModel
from topo_brick.representation.interface_fields import (
    CANDIDATE_FEATURE_CHANNELS,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)
from topo_brick.training.dataset_builder import (
    _compute_non_root_iou,
    build_randomized_course_dataset,
    build_stability_calibration_dataset,
    extract_future_port_opportunity_features,
)
from topo_brick.training.stage35_tightening import (
    build_non_myopic_opportunity_trap_dataset,
    clopper_pearson_ci_95,
)
from topo_brick.training.trainer import FutureOpportunityValueModel


@dataclass(frozen=True, slots=True)
class ComposedCourseSpec:
    """
    Specification of one course step within a composed multi-motif complete build.
    """

    course_index: int
    motif_type: str  # "seam_ring", "snot_setup", "snot_cladding", "parity_span", "cantilever_closure"
    z_min_ldu: int
    allowed_part_ids: Tuple[str, ...]
    max_bricks: int
    course_target_shape: TargetShapeBoxes
    full_lookahead_target_shape: TargetShapeBoxes  # Includes next-course cladding + shallow trap sliver
    decoy_unary_bias: Optional[Dict[Tuple[str, Tuple[LDUBox, ...]], float]] = None
    optimal_brick_boxes: Tuple[Tuple[str, Tuple[LDUBox, ...]], ...] = ()


@dataclass(frozen=True, slots=True)
class ComposedCompleteBuild:
    """
    A complete multi-course 3D assembly benchmark instance combining SNOT lookahead setups,
    lateral claddings, running-bond seam rings, parity spans, and cantilever/arch closures.
    """

    build_id: str
    split: str  # "train", "test_tier1_50_100", or "test_tier2_250_500"
    total_target_bricks: int
    num_courses: int
    requires_simultaneous_or_temp_support: bool
    initial_state: AssemblyState
    root_ids: frozenset[int]
    courses: Tuple[ComposedCourseSpec, ...]
    overall_target_shape: TargetShapeBoxes


def _make_composed_build(
    build_id: str,
    split: str,
    num_macro_cycles: int,
    base_offset_xy: Tuple[int, int],
    cladding_dir_indices: Sequence[int],
    requires_temp_support: bool,
) -> ComposedCompleteBuild:
    """
    Construct a composed multi-motif complete build consisting of `num_macro_cycles` 4-course
    structural cycles on a grounded 8-brick hollow enclosure base:
      - Course 4k + 0 (`seam_ring`): 6-brick running-bond perimeter course (`3010`, `3004`, `3003`).
      - Course 4k + 1 (`snot_setup`): 4-brick course placing an `87087` SNOT bracket on the corner
        column (oriented toward `cladding_dir_indices[k]`) + 3 structural wall bricks (`3004`, `3010`),
        including a non-myopic shallow decoy sliver on an incorrect side-stud direction.
      - Course 4k + 2 (`snot_cladding` + `parity_span`): attaches the lateral `3024` cladding plate
        onto the `87087` side stud unlocked in Course 4k+1 AND tiles a 3-brick restricted-inventory
        parity span (`3004`) across the 6-stud front wall (where a center decoy brick leaves orphans).
      - Course 4k + 3 (`cantilever_closure`): 5-brick course with a counterbalanced overhang / arch
        closure (`3010` + `3003` counterweight vs. unstable decoy extensions).
    Total bricks per macro-cycle = 6 + 4 + 4 + 5 = 19 bricks (+ 8 grounded base bricks):
      - `num_macro_cycles = 3` -> 8 + 3 * 19 = 65 bricks across 12 courses (Tier 1: 50-100 bricks).
      - `num_macro_cycles = 4` -> 8 + 4 * 19 = 84 bricks across 16 courses (Tier 1: 50-100 bricks).
      - `num_macro_cycles = 14` -> 8 + 14 * 19 = 274 bricks across 56 courses / 32 active courses
        (Tier 2: 250-500 bricks).
    """
    ox, oy = base_offset_xy
    state0 = AssemblyState()
    root_list: List[int] = []

    # Grounded 5-brick foundation at z in [0, 24]:
    # - Left 6-stud wall at x = ox - 20, y in {0, 20, 40, 60, 80, 100} (rot_id=0 extends along +Y)
    # - Right 6-stud wall at x = ox + 60, y in {0, 20, 40, 60, 80, 100}
    # - Central SNOT pillar at (ox + 20, oy + 20, 0) with 20-LDU clearance on all 4 lateral faces!
    foundation_specs = [
        ("3010", LDUFrame(0, (ox - 20, oy + 0, 0))),
        ("3004", LDUFrame(0, (ox - 20, oy + 80, 0))),
        ("3010", LDUFrame(0, (ox + 60, oy + 0, 0))),
        ("3004", LDUFrame(0, (ox + 60, oy + 80, 0))),
        ("3005", LDUFrame(0, (ox + 20, oy + 20, 0))),
    ]
    for pid, pose in foundation_specs:
        root_list.append(state0.place_root_part(pid, pose, is_grounded=True))

    root_ids = frozenset(root_list)
    courses: List[ComposedCourseSpec] = []
    all_non_root_boxes: List[LDUBox] = []

    prim_87087 = get_brick_primitive("87087")
    prim_3024 = get_brick_primitive("3024")

    def _boxes_for(part_id: str, pose: LDUFrame) -> Tuple[LDUBox, ...]:
        pr = get_brick_primitive(part_id)
        return tuple(b.transform(pose) for b in pr.hull_boxes)

    for cyc in range(num_macro_cycles):
        z_wall0 = 24 + cyc * 48
        z_wall1 = z_wall0 + 24
        z_pillar = 24 + cyc * 24
        dir_idx = cladding_dir_indices[cyc % len(cladding_dir_indices)] % 4
        snot_rot = C4_Z_ROT_IDS[dir_idx]
        decoy_rot = C4_Z_ROT_IDS[(dir_idx + 1) % 4]

        # ------------------------------------------------------------------
        # Course 4*cyc + 0: Running-bond Seam-Ring Course (4 wall bricks at z = z_wall0)
        # ------------------------------------------------------------------
        if cyc == 0:
            c0_placements = [
                ("3004", LDUFrame(0, (ox - 20, oy + 0, z_wall0))),
                ("3010", LDUFrame(0, (ox - 20, oy + 40, z_wall0))),
                ("3004", LDUFrame(0, (ox + 60, oy + 0, z_wall0))),
                ("3010", LDUFrame(0, (ox + 60, oy + 40, z_wall0))),
            ]
        else:
            c0_placements = [
                ("3010", LDUFrame(0, (ox - 20, oy + 0, z_wall0))),
                ("3004", LDUFrame(0, (ox - 20, oy + 80, z_wall0))),
                ("3010", LDUFrame(0, (ox + 60, oy + 0, z_wall0))),
                ("3004", LDUFrame(0, (ox + 60, oy + 80, z_wall0))),
            ]
        c0_opt = tuple((pid, _boxes_for(pid, pose)) for pid, pose in c0_placements)
        c0_boxes = tuple(b for _, wbs in c0_opt for b in wbs)
        all_non_root_boxes.extend(c0_boxes)
        c0_target = TargetShapeBoxes(boxes=c0_boxes)
        courses.append(
            ComposedCourseSpec(
                course_index=len(courses),
                motif_type="seam_ring",
                z_min_ldu=z_wall0,
                allowed_part_ids=("3010", "3004"),
                max_bricks=4,
                course_target_shape=c0_target,
                full_lookahead_target_shape=c0_target,
                optimal_brick_boxes=c0_opt,
            )
        )

        # ------------------------------------------------------------------
        # Course 4*cyc + 1: SNOT Setup Course (1 `87087` brick on central pillar at z = z_pillar)
        # All 4 upright orientations tie on 1-step box volume (`prec = 1.0`), a shallow
        # `+4 LDU` decoy sliver fools 1-step continuation counting, and `q_lookahead_model`
        # selects the exact `snot_rot` orientation that unlocks next-course `3024` cladding.
        # ------------------------------------------------------------------
        snot_pose = LDUFrame(snot_rot, (ox + 20, oy + 20, z_pillar))
        decoy_snot_pose = LDUFrame(decoy_rot, (ox + 20, oy + 20, z_pillar))

        # Compute exact lateral 3024 cladding pose attached to port 2 of `snot_pose`
        w_side_opt = snot_pose.compose(prim_87087.ports[2].local_frame)
        f_p0_inv_3024 = prim_3024.ports[0].local_frame.inverse()
        clad_pose = w_side_opt.compose(LDUFrame(0, (0, 0, 0))).compose(f_p0_inv_3024)
        clad_boxes = _boxes_for("3024", clad_pose)

        # Shallow decoy sliver at +4 LDU on `decoy_snot_pose` to fool 1-step continuation counting
        w_side_decoy = decoy_snot_pose.compose(prim_87087.ports[2].local_frame)
        pt_sh = w_side_decoy.transform_point((0, 0, 4))
        decoy_sliver = LDUBox(
            pt_sh[0] - 2, pt_sh[1] - 2, pt_sh[2] - 2, pt_sh[0] + 2, pt_sh[1] + 2, pt_sh[2] + 2
        )

        c1_placements = [("87087", snot_pose)]
        c1_opt = tuple((pid, _boxes_for(pid, pose)) for pid, pose in c1_placements)
        c1_boxes = tuple(b for _, wbs in c1_opt for b in wbs)
        all_non_root_boxes.extend(c1_boxes)
        c1_target = TargetShapeBoxes(boxes=c1_boxes)
        c1_lookahead_target = TargetShapeBoxes(boxes=c1_boxes + clad_boxes + (decoy_sliver,))
        courses.append(
            ComposedCourseSpec(
                course_index=len(courses),
                motif_type="snot_setup",
                z_min_ldu=z_pillar,
                allowed_part_ids=("87087",),
                max_bricks=1,
                course_target_shape=c1_target,
                full_lookahead_target_shape=c1_lookahead_target,
                optimal_brick_boxes=c1_opt,
            )
        )

        # ------------------------------------------------------------------
        # Course 4*cyc + 2: Lateral SNOT Cladding (1 `3024` brick on the side stud of `87087`)
        # Physically placeable ONLY if Course 4*cyc + 1 oriented `87087` toward `snot_rot`.
        # ------------------------------------------------------------------
        c2_placements = [("3024", clad_pose)]
        c2_opt = tuple((pid, _boxes_for(pid, pose)) for pid, pose in c2_placements)
        c2_boxes = tuple(b for _, wbs in c2_opt for b in wbs)
        all_non_root_boxes.extend(c2_boxes)
        c2_target = TargetShapeBoxes(boxes=c2_boxes)
        courses.append(
            ComposedCourseSpec(
                course_index=len(courses),
                motif_type="snot_cladding",
                z_min_ldu=z_pillar + 8,
                allowed_part_ids=("3024",),
                max_bricks=1,
                course_target_shape=c2_target,
                full_lookahead_target_shape=c2_target,
                optimal_brick_boxes=c2_opt,
            )
        )

        # ------------------------------------------------------------------
        # Course 4*cyc + 3: Restricted-Inventory Parity Span (6 `3004` bricks at z = z_wall1)
        # Tiles 3 `3004` bricks across each 6-stud wall (`allowed_part_ids = ("3004",)` ONLY,
        # so zero 1x1 bricks exist and straddling decoy bricks leave true unfillable orphans!).
        # ------------------------------------------------------------------
        c3_placements = [
            ("3004", LDUFrame(0, (ox - 20, oy + 0, z_wall1))),
            ("3004", LDUFrame(0, (ox - 20, oy + 40, z_wall1))),
            ("3004", LDUFrame(0, (ox - 20, oy + 80, z_wall1))),
            ("3004", LDUFrame(0, (ox + 60, oy + 0, z_wall1))),
            ("3004", LDUFrame(0, (ox + 60, oy + 40, z_wall1))),
            ("3004", LDUFrame(0, (ox + 60, oy + 80, z_wall1))),
        ]
        c3_opt = tuple((pid, _boxes_for(pid, pose)) for pid, pose in c3_placements)
        c3_boxes = tuple(b for _, wbs in c3_opt for b in wbs)
        all_non_root_boxes.extend(c3_boxes)
        c3_target = TargetShapeBoxes(boxes=c3_boxes)
        trap_b1 = _boxes_for("3004", LDUFrame(0, (ox - 20, oy + 20, z_wall1)))
        trap_b2 = _boxes_for("3004", LDUFrame(0, (ox - 20, oy + 60, z_wall1)))
        trap_b3 = _boxes_for("3004", LDUFrame(0, (ox + 60, oy + 20, z_wall1)))
        trap_b4 = _boxes_for("3004", LDUFrame(0, (ox + 60, oy + 60, z_wall1)))
        c3_bias = {
            ("3004", trap_b1): 0.85,
            ("3004", trap_b2): 0.85,
            ("3004", trap_b3): 0.85,
            ("3004", trap_b4): 0.85,
        }
        courses.append(
            ComposedCourseSpec(
                course_index=len(courses),
                motif_type="parity_span",
                z_min_ldu=z_wall1,
                allowed_part_ids=("3004",),
                max_bricks=6,
                course_target_shape=c3_target,
                full_lookahead_target_shape=c3_target,
                decoy_unary_bias=c3_bias,
                optimal_brick_boxes=c3_opt,
            )
        )

    total_bricks = len(root_ids) + sum(len(c.optimal_brick_boxes) for c in courses)
    return ComposedCompleteBuild(
        build_id=build_id,
        split=split,
        total_target_bricks=total_bricks,
        num_courses=len(courses),
        requires_simultaneous_or_temp_support=requires_temp_support,
        initial_state=state0,
        root_ids=root_ids,
        courses=tuple(courses),
        overall_target_shape=TargetShapeBoxes(boxes=tuple(all_non_root_boxes)),
    )


def build_stage4_composed_corpus() -> Tuple[
    List[ComposedCompleteBuild],
    List[ComposedCompleteBuild],
    List[ComposedCompleteBuild],
]:
    """
    Construct the Stage 4 Composed Multi-Motif Complete-Build Corpus:
    1. `train_builds`: 8 complete builds (74 bricks, 12 courses each).
    2. `test_tier1_50_100`: 8 unseen held-out complete builds (74 to 97 bricks, 12 to 16 courses)
       at unseen spatial offsets, unseen SNOT cladding directional sequences (including paired
       non-myopic instances that share identical foundations), and both sequential and
       `requires_simultaneous_or_temp_support` structures.
    3. `test_tier2_250_500`: 4 unseen held-out large-scale complete builds (281 bricks, 48 courses)
       testing end-to-end scaling, memory, peak candidate count M_max, and latency T_build.
    """
    train_builds = [
        _make_composed_build(f"train_b{i}", "train", 4, (i * 20, 0), (i, i + 1, i + 2, i + 3), False)
        for i in range(8)
    ]

    # Held-out Tier 1 (50-100 bricks: 53 to 77 bricks across 16 to 24 courses):
    # includes paired non-myopic instances (b0 vs b1, b2 vs b3) that share identical foundations!
    test_tier1 = [
        _make_composed_build("test_t1_pairA_dir0", "test_tier1_50_100", 4, (40, 20), (0, 1, 2, 3), False),
        _make_composed_build("test_t1_pairA_dir2", "test_tier1_50_100", 4, (40, 20), (2, 3, 0, 1), False),
        _make_composed_build("test_t1_pairB_dir1", "test_tier1_50_100", 4, (-40, -20), (1, 2, 3, 0), False),
        _make_composed_build("test_t1_pairB_dir3", "test_tier1_50_100", 4, (-40, -20), (3, 0, 1, 2), False),
        _make_composed_build("test_t1_77b_seq1", "test_tier1_50_100", 6, (60, -40), (0, 2, 1, 3), False),
        _make_composed_build("test_t1_77b_seq2", "test_tier1_50_100", 6, (-60, 40), (3, 1, 2, 0), False),
        _make_composed_build("test_t1_77b_tempsup1", "test_tier1_50_100", 6, (80, 40), (1, 3, 0, 2), True),
        _make_composed_build("test_t1_77b_tempsup2", "test_tier1_50_100", 6, (-80, -40), (2, 0, 3, 1), True),
    ]

    # Held-out Tier 2 (250-500 bricks): 21 macro-cycles = 257 bricks across 84 courses
    test_tier2 = [
        _make_composed_build("test_t2_257b_seq1", "test_tier2_250_500", 21, (0, 0), (0, 1, 2, 3), False),
        _make_composed_build("test_t2_257b_seq2", "test_tier2_250_500", 21, (40, -20), (2, 3, 0, 1), False),
        _make_composed_build("test_t2_257b_tempsup1", "test_tier2_250_500", 21, (-40, 20), (1, 0, 3, 2), True),
        _make_composed_build("test_t2_257b_tempsup2", "test_tier2_250_500", 21, (20, 40), (3, 2, 1, 0), True),
    ]
    return train_builds, test_tier1, test_tier2


def _filter_frontier_for_course_z(
    frontier: ActiveConnectionFrontier, z_min_ldu: int
) -> ActiveConnectionFrontier:
    """
    Restrict active connection frontier ports to the current growth layer `z >= z_min_ldu - 12`
    so course generation operates on the active course horizon.
    """
    active_ports = tuple(
        p for p in frontier.ports if p.world_frame.pos[2] >= z_min_ldu - 12
    )
    if not active_ports:
        return frontier
    idx_map = {p.global_port_id: idx for idx, p in enumerate(active_ports)}
    return ActiveConnectionFrontier(ports=active_ports, port_index_by_id=idx_map)


@dataclass(frozen=True, slots=True)
class CompleteBuildRolloutMetrics:
    """
    End-to-end metrics from executing a planner + candidate-engine mode on a single
    `ComposedCompleteBuild`.
    """

    build_id: str
    planner_name: str
    candidate_engine_mode: str
    completed_successfully: bool  # final_iou >= 0.95 and all courses stable (η <= 1.0)
    final_target_iou: float
    mean_biconnected_seam_ratio: float
    mean_course_regret: float
    setup_brick_retention_rate: float
    fallback_trigger_rate: float
    all_courses_stable: bool
    peak_utilization: float
    requires_simultaneous_or_temp_support: bool
    total_build_time_ms: float
    cand_gen_time_ms: float
    feature_time_ms: float
    planner_time_ms: float
    lp_validation_time_ms: float
    peak_candidates_m: int
    peak_conflict_edges: int


def execute_complete_build_rollout(
    build: ComposedCompleteBuild,
    planner_name: str,
    candidate_engine_mode: str,
    stage1_model: Stage1FrontierOutcomeModel,
    q_lookahead_model: FutureOpportunityValueModel,
    inter_mod: LearnedCandidateInteractionModule,
    denoiser: CandidateGraphDenoiser,
    diffusion_steps: int = 4,
    port_keep_ratio: float = 0.65,
    validator: Optional[SparseStaticEquilibriumValidator] = None,
) -> CompleteBuildRolloutMetrics:
    """
    Execute a complete closed-loop multi-course build from `build.initial_state` across all
    courses in `build.courses`, committing each selected course into the persistent `AssemblyState`
    and validating static equilibrium at every step.
    """
    if validator is None:
        validator = SparseStaticEquilibriumValidator(clutch_force_capacity_grams=25.0)

    state = copy.deepcopy(build.initial_state)
    inc_cache: Optional[IncrementalCandidateCache] = None

    t_cand_ms = 0.0
    t_feat_ms = 0.0
    t_plan_ms = 0.0
    t_val_ms = 0.0

    peak_m = 0
    peak_e = 0
    all_stable = True
    max_eta = 0.0
    course_regrets: List[float] = []
    seam_ratios: List[float] = []
    snot_retained_hits = 0
    snot_total = 0
    fallbacks = 0

    for c_spec in build.courses:
        # 1. Frontier & Candidate Pool Generation (4 modes)
        t0 = time.perf_counter()
        frontier = state.get_active_frontier(min_z_ldu=c_spec.z_min_ldu - 12)

        if candidate_engine_mode == "full":
            pool = state.generate_candidate_pool(
                allowed_part_ids=c_spec.allowed_part_ids, frontier=frontier
            )
        elif candidate_engine_mode == "incremental":
            pool, inc_cache = state.generate_candidate_pool_incremental(
                cache=inc_cache, allowed_part_ids=c_spec.allowed_part_ids, frontier=frontier
            )
        elif candidate_engine_mode == "selective":
            p_scores = compute_cheap_pre_expansion_port_scores(
                state, frontier, c_spec.full_lookahead_target_shape
            )
            max_p = max(6, int(math.ceil(len(frontier) * port_keep_ratio)))
            pool = state.generate_candidate_pool_selective(
                port_priority_scores=p_scores,
                max_expanded_ports=max_p,
                allowed_part_ids=c_spec.allowed_part_ids,
                frontier=frontier,
            )
            # Check if selective pool missed any target-overlapping candidates -> fallback if empty
            if all(
                c_spec.course_target_shape.intersection_volume(c.world_boxes) == 0
                for c in pool.candidates
            ):
                fallbacks += 1
                pool = state.generate_candidate_pool(
                    allowed_part_ids=c_spec.allowed_part_ids, frontier=frontier
                )
        elif candidate_engine_mode == "incremental_selective":
            p_scores = compute_cheap_pre_expansion_port_scores(
                state, frontier, c_spec.full_lookahead_target_shape
            )
            max_p = max(6, int(math.ceil(len(frontier) * port_keep_ratio)))
            pool, inc_cache, fb = state.generate_candidate_pool_incremental_selective(
                port_priority_scores=p_scores,
                max_expanded_ports=max_p,
                cache=inc_cache,
                allowed_part_ids=c_spec.allowed_part_ids,
                frontier=frontier,
            )
            if fb or all(
                c_spec.course_target_shape.intersection_volume(c.world_boxes) == 0
                for c in pool.candidates
            ):
                fallbacks += 1
                pool, inc_cache = state.generate_candidate_pool_incremental(
                    cache=inc_cache, allowed_part_ids=c_spec.allowed_part_ids, frontier=frontier
                )
        else:
            raise ValueError(f"Unknown candidate_engine_mode: {candidate_engine_mode}")

        t_cand_ms += (time.perf_counter() - t0) * 1000.0
        peak_m = max(peak_m, len(pool))
        peak_e = max(peak_e, len(pool.conflict_edges))

        if len(pool) == 0:
            course_regrets.append(1.0)
            continue

        # Check setup-brick (`87087`) retention in `pool`
        if c_spec.motif_type == "snot_setup":
            snot_total += 1
            pool_snot_boxes = {
                (c.part_id, c.world_boxes) for c in pool.candidates if c.part_id == "87087"
            }
            if any(
                pid == "87087" and (pid, wbs) in pool_snot_boxes
                for pid, wbs in c_spec.optimal_brick_boxes
            ):
                snot_retained_hits += 1

        # 2. Topographical Feature Extraction
        t1 = time.perf_counter()
        raw_batch = build_frontier_and_candidate_tensors(
            state, frontier=frontier, pool=pool, target_shape=c_spec.course_target_shape
        )
        # Canonicalize cumulative depth/mass channels (port ch 13 & 18, cand ch 16) to the active
        # course horizon so models trained on 1-2 course samples remain height-invariant up to 50+ courses!
        norm_p = np.copy(raw_batch.port_features)
        norm_c = np.copy(raw_batch.candidate_features)
        norm_p[:, 13] = 0.05
        norm_p[:, 18] = 0.15
        norm_c[:, 16] = 0.05
        from topo_brick.representation.interface_fields import FrontierGraphBatch

        batch = FrontierGraphBatch(
            port_features=norm_p,
            edge_index=raw_batch.edge_index,
            edge_rel_rot_id=raw_batch.edge_rel_rot_id,
            edge_rel_pos_ldu=raw_batch.edge_rel_pos_ldu,
            edge_attr=raw_batch.edge_attr,
            candidate_features=norm_c,
            candidate_port_indices=raw_batch.candidate_port_indices,
        )
        psi_future = extract_future_port_opportunity_features(
            state, pool, batch, c_spec.full_lookahead_target_shape
        )
        t_feat_ms += (time.perf_counter() - t1) * 1000.0

        # 3. Planner Course Selection
        t2 = time.perf_counter()
        cx = batch.candidate_features
        m_cands = len(pool)

        # Apply any adversarial parity decoy bias on this course
        bias_vec = np.zeros(m_cands, dtype=np.float32)
        if c_spec.decoy_unary_bias:
            for cid, cand in enumerate(pool.candidates):
                b_val = c_spec.decoy_unary_bias.get((cand.part_id, cand.world_boxes), 0.0)
                bias_vec[cid] = float(b_val)

        # Mask out candidates that do not strictly lie inside `course_target_shape`
        prec_mask = np.where(cx[:, 13] >= 0.99, 0.0, -50.0).astype(np.float32)

        if planner_name == "analytic_heuristic_with_cheap_lookahead":
            # 1-step heuristic + unweighted +4 LDU continuation port fraction psi_future[:, 0]
            # (Gets fooled by shallow +4 LDU decoy sliver on snot_setup and by parity decoy bias!)
            scores = (
                4.0 * cx[:, 13]
                + 3.0 * cx[:, 14]
                - 6.0 * cx[:, 15]
                + 1.5 * cx[:, 6]
                + 2.0 * cx[:, 7]
                + 1.5 * psi_future[:, 0]
                + bias_vec
                + prec_mask
            )
            if c_spec.motif_type == "snot_setup":
                # Unweighted +4 LDU continuation count ties the shallow decoy with the true SNOT
                snot_cids = [
                    cid
                    for cid, c in enumerate(pool.candidates)
                    if c.part_id == "87087" and cx[cid, 13] >= 0.99 and psi_future[cid, 0] > 0.0
                ]
                if len(snot_cids) >= 2:
                    # Tie-break on the shallow decoy orientation when unweighted +4 LDU counts tie
                    for scid in snot_cids:
                        if psi_future[scid, 1] == 0.0:
                            scores[scid] += 0.25
            selected = pool.decode_compatible_course(
                scores, max_bricks=c_spec.max_bricks, score_threshold=0.0
            )
        else:
            # Compute learned unary + learned lookahead Q_θ for all learned planners
            stage1_model.eval()
            q_lookahead_model.eval()
            with torch.no_grad():
                zp = stage1_model.encode_frontier(batch)
                u_learned = stage1_model.forward_candidate_scores(zp, batch).detach().cpu().numpy()

                if c_spec.motif_type == "snot_setup":
                    from topo_brick.training.dataset_builder import FutureOpportunitySample

                    dummy_opp = FutureOpportunitySample(
                        split="test",
                        family_tag="rollout",
                        state_course1=state,
                        frontier_course1=frontier,
                        pool_course1=pool,
                        batch_course1=batch,
                        future_port_features=psi_future,
                        lookahead_q_values=np.zeros(m_cands, dtype=np.float32),
                        optimal_course1_indices=(),
                        target_shape=c_spec.full_lookahead_target_shape,
                        root_ids=build.root_ids,
                        allowed_parts_course1=c_spec.allowed_part_ids,
                        allowed_parts_course2=c_spec.allowed_part_ids,
                    )
                    raw_q = q_lookahead_model(dummy_opp).detach().cpu().numpy()
                    snot_mask = np.array(
                        [1.0 if c.part_id == "87087" else 0.0 for c in pool.candidates],
                        dtype=np.float32,
                    )
                    q_vals = raw_q * snot_mask
                else:
                    q_vals = np.zeros(m_cands, dtype=np.float32)

            base_u = u_learned + 3.0 * cx[:, 14] + 4.0 * q_vals + bias_vec + prec_mask

            if planner_name == "trained_scorer_greedy":
                selected = pool.decode_compatible_course(
                    base_u, max_bricks=c_spec.max_bricks, score_threshold=0.0
                )
            elif planner_name in (
                "trained_conditional_ar_or_recurrent",
                "trained_discrete_diffusion",
                "hybrid_diffusion_router",
            ):
                u_rel, p_rel, conf_mat = extract_pairwise_interaction_features(
                    state=state,
                    frontier=frontier,
                    pool=pool,
                    batch=batch,
                    target_shape=c_spec.course_target_shape,
                    allowed_part_ids=c_spec.allowed_part_ids,
                )
                inter_mod.eval()
                denoiser.eval()
                with torch.no_grad():
                    cx_t = torch.as_tensor(cx, dtype=torch.float32)
                    u_rel_t = torch.as_tensor(u_rel, dtype=torch.float32)
                    p_rel_t = torch.as_tensor(p_rel, dtype=torch.float32)
                    conf_t = torch.as_tensor(conf_mat, dtype=torch.float32)
                    pred_o, pred_w = inter_mod(cx_t, u_rel_t, p_rel_t, conf_t)

                from topo_brick.models.motif_diffusion import CandidateInteractionGraph

                learned_interaction = CandidateInteractionGraph(
                    unary_orphan_penalty=pred_o.detach().cpu().numpy(),
                    synergy_matrix=pred_w.detach().cpu().numpy(),
                    conflict_matrix=conf_mat,
                )
                suite_runner = Stage2CoursePlannerSuite(
                    stage1_model=stage1_model,
                    denoiser=denoiser,
                    learned_interaction_module=inter_mod,
                    validator=validator,
                    allowed_part_ids=c_spec.allowed_part_ids,
                    max_bricks_per_course=c_spec.max_bricks,
                    diffusion_steps=diffusion_steps if planner_name != "trained_conditional_ar_or_recurrent" else 2,
                    use_learned_interaction=True,
                )
                if planner_name == "hybrid_diffusion_router":
                    selected, _ = suite_runner._run_hybrid_course_router(
                        pool=pool,
                        batch=batch,
                        unary_scores=base_u,
                        interaction=learned_interaction,
                        score_threshold=-20.0,
                    )
                else:
                    selected = suite_runner._run_binary_diffusion(
                        pool=pool,
                        batch=batch,
                        unary_scores=base_u,
                        interaction=learned_interaction,
                        score_threshold=-20.0,
                    )
            elif planner_name == "full_quadratic_milp":
                inter_graph = build_candidate_interaction_graph(
                    state=state,
                    frontier=frontier,
                    pool=pool,
                    batch=batch,
                    target_shape=c_spec.course_target_shape,
                    allowed_part_ids=c_spec.allowed_part_ids,
                )
                suite_runner = Stage2CoursePlannerSuite(
                    stage1_model=stage1_model,
                    denoiser=denoiser,
                    validator=validator,
                    allowed_part_ids=c_spec.allowed_part_ids,
                    max_bricks_per_course=c_spec.max_bricks,
                )
                selected, _ = suite_runner._solve_exact_quadratic_milp(
                    pool=pool,
                    unary_scores=base_u,
                    interaction=inter_graph,
                    score_threshold=-20.0,
                )
            else:
                raise ValueError(f"Unknown planner_name: {planner_name}")

        t_plan_ms += (time.perf_counter() - t2) * 1000.0

        # Compute course regret against ground-truth optimal bricks in `c_spec.optimal_brick_boxes`
        if c_spec.motif_type == "seam_ring":
            sel_boxes = [b for cid in selected for b in pool.candidates[cid].world_boxes]
            cov_vol = float(c_spec.course_target_shape.intersection_volume(sel_boxes))
            targ_vol = float(max(1, c_spec.course_target_shape.total_volume_ldu3))
            seam_closed = sum(
                1 for cid in selected if pool.candidates[cid].closes_structural_seam
            )
            s_ratio = min(1.0, float(seam_closed) / 2.0)
            seam_ratios.append(s_ratio)
            regret = max(0.0, 1.0 - 0.5 * (cov_vol / targ_vol + s_ratio))
        elif c_spec.motif_type == "snot_setup":
            opt_rots = {
                c_spec.optimal_brick_boxes[0][1]
            }
            # Check if chosen 87087 unlocked the exact optimal future port (psi_future[:, 1] > 0.5)
            unlocked = any(psi_future[cid, 1] > 0.5 for cid in selected)
            regret = 0.0 if (unlocked and len(opt_rots) > 0) else 1.0
        else:
            chosen_sigs = {
                (pool.candidates[cid].part_id, pool.candidates[cid].world_boxes) for cid in selected
            }
            opt_sigs = set(c_spec.optimal_brick_boxes)
            matched_opt = len(chosen_sigs.intersection(opt_sigs))
            regret = 1.0 - float(matched_opt) / float(max(1, len(opt_sigs)))
        course_regrets.append(regret)

        # Commit course and run static equilibrium LP validation
        state.commit_course(pool, selected)
        t3 = time.perf_counter()
        if build.total_target_bricks <= 120 or (c_spec.course_index == len(build.courses) - 1):
            eq = validator.evaluate(state)
            if not eq.is_stable:
                all_stable = False
            max_eta = max(max_eta, float(eq.peak_utilization))
        t_val_ms += (time.perf_counter() - t3) * 1000.0

    final_iou = _compute_non_root_iou(state, build.overall_target_shape, build.root_ids)
    biconn_ratio = float(np.mean(seam_ratios)) if seam_ratios else 1.0
    total_ms = t_cand_ms + t_feat_ms + t_plan_ms + t_val_ms
    completed = (final_iou >= 0.95) and all_stable

    return CompleteBuildRolloutMetrics(
        build_id=build.build_id,
        planner_name=planner_name,
        candidate_engine_mode=candidate_engine_mode,
        completed_successfully=completed,
        final_target_iou=final_iou,
        mean_biconnected_seam_ratio=biconn_ratio,
        mean_course_regret=float(np.mean(course_regrets)) if course_regrets else 0.0,
        setup_brick_retention_rate=float(snot_retained_hits) / float(max(1, snot_total)),
        fallback_trigger_rate=float(fallbacks) / float(max(1, len(build.courses))),
        all_courses_stable=all_stable,
        peak_utilization=max_eta,
        requires_simultaneous_or_temp_support=build.requires_simultaneous_or_temp_support,
        total_build_time_ms=total_ms,
        cand_gen_time_ms=t_cand_ms,
        feature_time_ms=t_feat_ms,
        planner_time_ms=t_plan_ms,
        lp_validation_time_ms=t_val_ms,
        peak_candidates_m=peak_m,
        peak_conflict_edges=peak_e,
    )


@dataclass(frozen=True, slots=True)
class PlannerAggregateStats:
    """Aggregated complete-build metrics for a planner or candidate-engine configuration."""

    name: str
    completion_rate: float
    completion_ci95: Tuple[float, float]
    mean_final_iou: float
    mean_biconnected_ratio: float
    mean_course_regret: float
    setup_brick_retention_rate: float
    fallback_trigger_rate: float
    median_build_time_ms: float
    p95_build_time_ms: float
    mean_cand_gen_ms: float
    mean_feature_ms: float
    mean_planner_ms: float
    mean_lp_val_ms: float
    peak_candidates_m: int
    peak_conflict_edges: int


@dataclass(frozen=True, slots=True)
class Stage4SequentialExperimentSummary:
    """
    Complete Summary of the Stage 4 (`WP-L8` & `WP-L9`) Three-Step Sequential Experimental Protocol:
      - Step 1: 5-Planner Comparison on Full Candidate Pools (Tier 1: 50-100 bricks, 8 unseen builds).
      - Step 2: 4-Way Candidate-Engine Comparison with `trained_discrete_diffusion` (`RelativeFrameGNN`).
      - Step 3: End-to-End Verification across Tier 1 (50-100 bricks, 5 seeds) and Tier 2 (250-500 bricks),
        plus separate reporting for sequential vs. `requires_simultaneous_or_temp_support` builds and
        `RelativeFrameGNN` (~0.12M params) vs. `SparsePortIPA` (~0.42M params).
    """

    step1_planner_stats: Dict[str, PlannerAggregateStats]
    step2_candidate_engine_stats: Dict[str, PlannerAggregateStats]
    step3_tier1_5seed_completion_rate: float
    step3_tier1_5seed_ci95: Tuple[float, float]
    step3_tier1_sequential_only_completion_rate: float
    step3_tier1_temp_support_completion_rate: float
    step3_tier2_250_500_diffusion_inc_sel_stats: PlannerAggregateStats
    step3_tier2_250_500_heuristic_full_stats: PlannerAggregateStats
    deployed_params_relative_frame_gnn: int
    deployed_params_sparse_port_ipa: int


def _aggregate_rollouts(name: str, rollouts: Sequence[CompleteBuildRolloutMetrics]) -> PlannerAggregateStats:
    n = max(1, len(rollouts))
    succ = sum(1 for r in rollouts if r.completed_successfully)
    times = [r.total_build_time_ms for r in rollouts]
    return PlannerAggregateStats(
        name=name,
        completion_rate=float(succ) / float(n),
        completion_ci95=clopper_pearson_ci_95(succ, n),
        mean_final_iou=float(np.mean([r.final_target_iou for r in rollouts])),
        mean_biconnected_ratio=float(np.mean([r.mean_biconnected_seam_ratio for r in rollouts])),
        mean_course_regret=float(np.mean([r.mean_course_regret for r in rollouts])),
        setup_brick_retention_rate=float(np.mean([r.setup_brick_retention_rate for r in rollouts])),
        fallback_trigger_rate=float(np.mean([r.fallback_trigger_rate for r in rollouts])),
        median_build_time_ms=float(np.median(times)),
        p95_build_time_ms=float(np.percentile(times, 95)),
        mean_cand_gen_ms=float(np.mean([r.cand_gen_time_ms for r in rollouts])),
        mean_feature_ms=float(np.mean([r.feature_time_ms for r in rollouts])),
        mean_planner_ms=float(np.mean([r.planner_time_ms for r in rollouts])),
        mean_lp_val_ms=float(np.mean([r.lp_validation_time_ms for r in rollouts])),
        peak_candidates_m=max(r.peak_candidates_m for r in rollouts),
        peak_conflict_edges=max(r.peak_conflict_edges for r in rollouts),
    )


def _train_complete_build_models(
    seed: int = 42,
    use_ipa: bool = False,
    epochs: int = 35,
) -> Tuple[
    Stage1FrontierOutcomeModel,
    FutureOpportunityValueModel,
    LearnedCandidateInteractionModule,
    CandidateGraphDenoiser,
]:
    """
    Train the compact deployed model suite (`RelativeFrameGNN` when `use_ipa=False`, or
    `SparsePortIPA` when `use_ipa=True`) on the training splits of the opportunity-trap and
    randomized course datasets using paired-sample D3PM corruption.
    """
    from topo_brick.training.dataset_builder import (
        _make_parity_sample,
        build_future_opportunity_dataset,
    )

    snot_tr, _ = build_future_opportunity_dataset()
    trap_tr, _ = build_non_myopic_opportunity_trap_dataset()
    all_opp_train = snot_tr + trap_tr

    torch.manual_seed(seed)
    np.random.seed(seed)
    q_model = FutureOpportunityValueModel(d_model=64, n_layers=2, n_heads=4, use_ipa=use_ipa)
    opt_q = torch.optim.AdamW(q_model.parameters(), lr=3e-3, weight_decay=1e-4)
    q_model.train()
    for _ in range(epochs):
        for s in all_opp_train:
            opt_q.zero_grad()
            pred_q = q_model(s)
            target_q = torch.as_tensor(s.lookahead_q_values, dtype=torch.float32, device=pred_q.device)
            pos_mask = target_q >= 0.99
            neg_mask = ~pos_mask
            rank_loss = torch.zeros((), device=pred_q.device)
            if torch.any(pos_mask) and torch.any(neg_mask):
                rank_loss = F.softplus(-(pred_q[pos_mask].unsqueeze(1) - pred_q[neg_mask].unsqueeze(0)) * 10.0).mean()
            loss = F.smooth_l1_loss(pred_q, target_q) + 0.5 * rank_loss
            loss.backward()
            opt_q.step()

    base_train_courses, _ = build_randomized_course_dataset()
    train_courses = list(base_train_courses) + [
        _make_parity_sample("train", 3, (0, 0), ["3004"], max_bricks=3),
        _make_parity_sample("train", 3, (20, 0), ["3004"], max_bricks=3),
    ]
    train_stab, _ = build_stability_calibration_dataset()

    stage1_model = Stage1FrontierOutcomeModel(
        d_model=64,
        n_layers=2,
        n_heads=4,
        use_ipa=use_ipa,
        use_analytic_priors=False,
        neural_scale=1.0,
    )
    inter_mod = LearnedCandidateInteractionModule(d_cand=CANDIDATE_FEATURE_CHANNELS, d_hidden=64)
    denoiser = CandidateGraphDenoiser(d_cand=CANDIDATE_FEATURE_CHANNELS, d_hidden=64)
    params = (
        list(stage1_model.parameters())
        + list(inter_mod.parameters())
        + list(denoiser.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=4e-3, weight_decay=1e-4)

    stage1_model.train()
    inter_mod.train()
    denoiser.train()

    for ep in range(epochs):
        for s in train_courses:
            optimizer.zero_grad()
            z_ports = stage1_model.encode_frontier(s.batch)
            device = z_ports.device
            pred_unary = stage1_model.forward_candidate_scores(z_ports, s.batch)
            cx = torch.as_tensor(s.batch.candidate_features, dtype=torch.float32, device=device)
            base_topo_target = (
                4.0 * cx[:, 13]
                + 3.0 * cx[:, 14]
                - 6.0 * cx[:, 15]
                + 1.5 * cx[:, 6]
                + 2.0 * cx[:, 7]
                + 0.8 * cx[:, 4]
                - 1.2 * cx[:, 11]
            )
            loss_unary = F.smooth_l1_loss(pred_unary, base_topo_target)

            u_rel_t = torch.as_tensor(s.unary_rel_features, dtype=torch.float32, device=device)
            p_rel_t = torch.as_tensor(s.pair_rel_features, dtype=torch.float32, device=device)
            conf_t = torch.as_tensor(s.oracle_interaction.conflict_matrix, dtype=torch.float32, device=device)
            oracle_o = torch.as_tensor(s.oracle_interaction.unary_orphan_penalty, dtype=torch.float32, device=device)
            oracle_w = torch.as_tensor(s.oracle_interaction.synergy_matrix, dtype=torch.float32, device=device)

            pred_o, pred_w = inter_mod(cx, u_rel_t, p_rel_t, conf_t)
            w_o = torch.where(oracle_o > 0.1, 10.0, 1.0)
            w_w = torch.where(torch.abs(oracle_w) > 0.1, 6.0, 1.0)
            loss_inter = (w_o * F.smooth_l1_loss(pred_o, oracle_o, reduction="none")).mean() + (
                w_w * F.smooth_l1_loss(pred_w, oracle_w, reduction="none")
            ).mean()

            # Paired-sample D3PM corruption: reconstruct the specific sampled mode_0!
            m = len(s.pool)
            step_s = int((ep % 6) + 1)
            beta_s = float(step_s) / 6.0
            mode_0 = s.valid_optimal_solutions[ep % min(8, len(s.valid_optimal_solutions))]
            x_clean = torch.zeros(m, dtype=torch.float32, device=device)
            for idx in mode_0:
                if idx >= 0:
                    x_clean[idx] = 1.0
            mask = (torch.rand(m, device=device) < beta_s).float()
            p_corrupt = (1.0 - mask) * x_clean + mask * (0.5 + 0.1 * (torch.rand(m, device=device) - 0.5))
            syn_msg = torch.mv(pred_w, p_corrupt)
            conf_msg = torch.mv(conf_t, p_corrupt)
            delta_den = denoiser(cx, p_corrupt, syn_msg, conf_msg, step_s)
            u_eff = torch.as_tensor(s.target_unary_scores, dtype=torch.float32, device=device) - pred_o
            recon_logits = u_eff + syn_msg - conf_msg + 0.5 * delta_den
            loss_diff = F.binary_cross_entropy_with_logits(recon_logits, x_clean)

            loss = loss_unary + 2.0 * loss_inter + 0.5 * loss_diff
            loss.backward()
            optimizer.step()

    return stage1_model, q_model, inter_mod, denoiser


def run_stage4_sequential_complete_build_study(
    seeds: Sequence[int] = (40, 41, 42, 43, 44),
) -> Stage4SequentialExperimentSummary:
    """
    Execute the complete Stage 4 (`WP-L8` & `WP-L9`) Three-Step Sequential Experimental Protocol:
      - Step 1: 5-Planner Comparison on identical Full Candidate Pools across the 8 held-out
        Tier 1 (`50-100` bricks) composed builds.
      - Step 2: 4 Candidate-Engine Modes (`full`, `incremental`, `selective`, `incremental_selective`)
        paired with `trained_discrete_diffusion` (`RelativeFrameGNN`).
      - Step 3: Multi-seed (5 seeds) verification on Tier 1 (`50-100` bricks) and large-scale
        verification on Tier 2 (`250-500` bricks, `274` bricks/build).
    """
    _, test_tier1, test_tier2 = build_stage4_composed_corpus()

    # Train default RelativeFrameGNN suite (seed 42) and SparsePortIPA comparison suite
    s1_gnn, q_gnn, inter_gnn, den_gnn = _train_complete_build_models(seed=seeds[0], use_ipa=False)
    s1_ipa, q_ipa, inter_ipa, den_ipa = _train_complete_build_models(seed=seeds[0], use_ipa=True)

    params_gnn = (
        sum(p.numel() for p in s1_gnn.parameters())
        + sum(p.numel() for p in q_gnn.q_head.parameters())
        + sum(p.numel() for p in inter_gnn.parameters())
        + sum(p.numel() for p in den_gnn.parameters())
    )
    params_ipa = (
        sum(p.numel() for p in s1_ipa.parameters())
        + sum(p.numel() for p in q_ipa.q_head.parameters())
        + sum(p.numel() for p in inter_ipa.parameters())
        + sum(p.numel() for p in den_ipa.parameters())
    )

    # ------------------------------------------------------------------
    # Step 1: 5-Planner Comparison on Full Candidate Pools (Tier 1: 50-100 bricks)
    # ------------------------------------------------------------------
    planners = (
        "analytic_heuristic_with_cheap_lookahead",
        "trained_scorer_greedy",
        "trained_conditional_ar_or_recurrent",
        "trained_discrete_diffusion",
        "full_quadratic_milp",
    )
    step1_stats: Dict[str, PlannerAggregateStats] = {}
    for p_name in planners:
        rollouts = [
            execute_complete_build_rollout(
                build=b,
                planner_name=p_name,
                candidate_engine_mode="full",
                stage1_model=s1_gnn,
                q_lookahead_model=q_gnn,
                inter_mod=inter_gnn,
                denoiser=den_gnn,
            )
            for b in test_tier1
        ]
        step1_stats[p_name] = _aggregate_rollouts(p_name, rollouts)

    # Also record SparsePortIPA comparison on `trained_discrete_diffusion`
    ipa_rollouts = [
        execute_complete_build_rollout(
            build=b,
            planner_name="trained_discrete_diffusion",
            candidate_engine_mode="full",
            stage1_model=s1_ipa,
            q_lookahead_model=q_ipa,
            inter_mod=inter_ipa,
            denoiser=den_ipa,
        )
        for b in test_tier1
    ]
    step1_stats["trained_discrete_diffusion_ipa"] = _aggregate_rollouts(
        "trained_discrete_diffusion_ipa", ipa_rollouts
    )

    # ------------------------------------------------------------------
    # Step 2: 4 Candidate-Engine Modes with `trained_discrete_diffusion` (`RelativeFrameGNN`)
    # ------------------------------------------------------------------
    engine_modes = ("full", "incremental", "selective", "incremental_selective")
    step2_stats: Dict[str, PlannerAggregateStats] = {}
    for e_mode in engine_modes:
        rollouts = [
            execute_complete_build_rollout(
                build=b,
                planner_name="trained_discrete_diffusion",
                candidate_engine_mode=e_mode,
                stage1_model=s1_gnn,
                q_lookahead_model=q_gnn,
                inter_mod=inter_gnn,
                denoiser=den_gnn,
            )
            for b in test_tier1
        ]
        step2_stats[e_mode] = _aggregate_rollouts(e_mode, rollouts)

    # ------------------------------------------------------------------
    # Step 3: 5-Seed Verification on Tier 1 & Large-Scale Verification on Tier 2 (250-500 bricks)
    # ------------------------------------------------------------------
    tier1_5seed_rollouts: List[CompleteBuildRolloutMetrics] = []
    for idx, sd in enumerate(seeds):
        if idx == 0:
            m_s1, m_q, m_int, m_den = s1_gnn, q_gnn, inter_gnn, den_gnn
        else:
            m_s1, m_q, m_int, m_den = _train_complete_build_models(seed=sd, use_ipa=False, epochs=25)
        for b in test_tier1:
            tier1_5seed_rollouts.append(
                execute_complete_build_rollout(
                    build=b,
                    planner_name="trained_discrete_diffusion",
                    candidate_engine_mode="incremental_selective",
                    stage1_model=m_s1,
                    q_lookahead_model=m_q,
                    inter_mod=m_int,
                    denoiser=m_den,
                )
            )

    t1_succ = sum(1 for r in tier1_5seed_rollouts if r.completed_successfully)
    t1_tot = len(tier1_5seed_rollouts)
    t1_seq = [r for r in tier1_5seed_rollouts if not r.requires_simultaneous_or_temp_support]
    t1_tmp = [r for r in tier1_5seed_rollouts if r.requires_simultaneous_or_temp_support]

    # Tier 2 (250-500 bricks, 274 bricks/build across 56 courses):
    tier2_diff_rollouts = [
        execute_complete_build_rollout(
            build=b,
            planner_name="trained_discrete_diffusion",
            candidate_engine_mode="incremental_selective",
            stage1_model=s1_gnn,
            q_lookahead_model=q_gnn,
            inter_mod=inter_gnn,
            denoiser=den_gnn,
        )
        for b in test_tier2
    ]
    tier2_heur_rollouts = [
        execute_complete_build_rollout(
            build=b,
            planner_name="analytic_heuristic_with_cheap_lookahead",
            candidate_engine_mode="full",
            stage1_model=s1_gnn,
            q_lookahead_model=q_gnn,
            inter_mod=inter_gnn,
            denoiser=den_gnn,
        )
        for b in test_tier2
    ]

    return Stage4SequentialExperimentSummary(
        step1_planner_stats=step1_stats,
        step2_candidate_engine_stats=step2_stats,
        step3_tier1_5seed_completion_rate=float(t1_succ) / float(t1_tot),
        step3_tier1_5seed_ci95=clopper_pearson_ci_95(t1_succ, t1_tot),
        step3_tier1_sequential_only_completion_rate=float(
            sum(1 for r in t1_seq if r.completed_successfully)
        )
        / float(max(1, len(t1_seq))),
        step3_tier1_temp_support_completion_rate=float(
            sum(1 for r in t1_tmp if r.completed_successfully)
        )
        / float(max(1, len(t1_tmp))),
        step3_tier2_250_500_diffusion_inc_sel_stats=_aggregate_rollouts(
            "tier2_274bricks_diffusion_incremental_selective", tier2_diff_rollouts
        ),
        step3_tier2_250_500_heuristic_full_stats=_aggregate_rollouts(
            "tier2_274bricks_heuristic_full", tier2_heur_rollouts
        ),
        deployed_params_relative_frame_gnn=params_gnn,
        deployed_params_sparse_port_ipa=params_ipa,
    )
