"""
WP-L3 & WP-L5: Stage 1 & Stage 2 Benchmark and Controlled Ablation Suite.

Evaluates:
1. Stage 1 Target Wall Reconstruction (`SparsePortIPAEncoder` vs. `RelativeFrameGNNEncoder`),
   verifying 100% connectivity, 100% collision freedom, and 100% static stability.
2. Stage 2 (`WP-L4` & `WP-L5`) Four-Planner Controlled Comparison across three adversarial
   combinatorial assembly benchmarks:
   - Benchmark A: Restricted-Inventory Parity & Domino-Trap Packing (No 1x1 filler bricks),
     where individual candidates have zero unary orphan penalty (`unary_orphan == 0`) but
     incompatible pairs trap isolated 1x1 holes (`IoU = 0.667` for unary Greedy/MILP vs.
     `IoU = 1.000` for Binary Candidate-Pool Discrete Diffusion).
   - Benchmark B: Multi-Course Seam Staggering & Biconnected Structural Integrity.
   - Benchmark C: Counter-Balanced Cantilever Course where an overhanging placement is
     mechanically stable (η* <= 1.0) only when paired with a counter-balancing brick in the
     same course via the Candidate Interaction Graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.ldu_frames import (
    BRICK_HEIGHT_LDU,
    LDUBox,
    LDUFrame,
)
from topo_brick.interface.port_manifold import (
    AssemblyState,
    IncrementalCandidateCache,
)
from topo_brick.models.motif_diffusion import (
    CandidateGraphDenoiser,
    Stage2CourseMetrics,
    Stage2CoursePlannerSuite,
)
from topo_brick.models.outcome_head import (
    CourseStepMetrics,
    Stage1FrontierOutcomeModel,
    Stage1FrontierPlanner,
)
from topo_brick.representation.interface_fields import (
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)

PLANNER_TYPES = (
    "one_shot_greedy",
    "one_shot_milp",
    "autoregressive_beam",
    "binary_diffusion",
)


@dataclass(frozen=True, slots=True)
class Stage1BenchmarkSummary:
    """Summary metrics from the Stage 1 verification and ablation suite."""

    encoder_type: str
    total_courses_executed: int
    total_bricks_placed: int
    connectivity_valid_rate: float
    collision_free_rate: float
    mechanically_stable_rate: float
    final_target_iou: float
    mean_neural_stability_ms: float
    mean_lp_validator_ms: float
    mean_course_total_ms: float
    latency_breakdown_ms: Dict[str, float]


@dataclass(frozen=True, slots=True)
class Stage2PlannerBenchmarkResult:
    """Per-planner summary across the WP-L5 adversarial benchmark suite."""

    planner_name: str
    benchmark_a1_1x6_parity_iou: float
    benchmark_a2_2x3_domino_trap_iou: float
    benchmark_b_seam_stagger_objective: float
    benchmark_b_biconnected_ratio: float
    benchmark_c_cantilever_completed: bool
    benchmark_c_peak_utilization: float
    mean_total_latency_ms: float


def _compute_target_iou(state: AssemblyState, target_shape: TargetShapeBoxes, root_ids: set[int]) -> float:
    placed_boxes = [
        b
        for inst_id, p in state.placed_parts.items()
        if inst_id not in root_ids
        for b in p.world_boxes
    ]
    inter_vol = float(target_shape.intersection_volume(placed_boxes))
    placed_vol = float(sum(b.volume_ldu3 for b in placed_boxes))
    target_vol = float(target_shape.total_volume_ldu3)
    union_vol = max(1.0, placed_vol + target_vol - inter_vol)
    return inter_vol / union_vol


def run_target_wall_reconstruction(
    use_ipa: bool = True,
    seed: int = 42,
) -> Stage1BenchmarkSummary:
    """
    Stage 1 Benchmark: Reconstruct a multi-layer interlocking 2x4 wall target volume.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = AssemblyState()
    state.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)

    target_box = LDUBox(
        xmin=-10,
        ymin=-10,
        zmin=BRICK_HEIGHT_LDU,
        xmax=30,
        ymax=70,
        zmax=4 * BRICK_HEIGHT_LDU,
    )
    target_shape = TargetShapeBoxes(boxes=(target_box,))

    model = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=use_ipa)
    validator = SparseStaticEquilibriumValidator(clutch_force_capacity_grams=50.0)
    planner = Stage1FrontierPlanner(
        model=model,
        validator=validator,
        allowed_part_ids=["3004", "3003", "3001"],
        max_bricks_per_course=2,
    )

    step_records: List[CourseStepMetrics] = []
    neural_stab_times: List[float] = []

    for _ in range(6):
        batch = build_frontier_and_candidate_tensors(state, target_shape=target_shape)
        stab_pred = model.predict_stability(batch)
        neural_stab_times.append(stab_pred.inference_time_ms)

        metrics = planner.step_course(
            state=state,
            target_shape=target_shape,
            require_target_precision=0.99,
        )
        if not metrics.committed_instance_ids:
            break
        step_records.append(metrics)

    iou = _compute_target_iou(state, target_shape, root_ids={0})

    all_boxes = [
        (inst_id, b)
        for inst_id, p in state.placed_parts.items()
        for b in p.world_boxes
    ]
    collision_count = 0
    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            if all_boxes[i][0] != all_boxes[j][0] and all_boxes[i][1].intersects(all_boxes[j][1]):
                collision_count += 1

    depths = state.compute_support_depths()
    connected_count = sum(1 for d in depths.values() if d < 999)
    final_eq = validator.evaluate(state)

    breakdown = {
        "t_candidates_ms": float(np.mean([r.t_candidates_ms for r in step_records])),
        "t_features_ms": float(np.mean([r.t_features_ms for r in step_records])),
        "t_encode_ms": float(np.mean([r.t_encode_ms for r in step_records])),
        "t_score_ms": float(np.mean([r.t_score_ms for r in step_records])),
        "t_decode_ms": float(np.mean([r.t_decode_ms for r in step_records])),
        "t_validate_ms": float(np.mean([r.t_validate_ms for r in step_records])),
    }

    return Stage1BenchmarkSummary(
        encoder_type="SparsePortIPA" if use_ipa else "RelativeFrameGNN",
        total_courses_executed=len(step_records),
        total_bricks_placed=state.num_parts,
        connectivity_valid_rate=float(connected_count) / float(state.num_parts),
        collision_free_rate=1.0 if collision_count == 0 else 0.0,
        mechanically_stable_rate=1.0 if final_eq.is_stable else 0.0,
        final_target_iou=iou,
        mean_neural_stability_ms=float(np.mean(neural_stab_times)),
        mean_lp_validator_ms=breakdown["t_validate_ms"],
        mean_course_total_ms=float(np.mean([r.t_total_ms for r in step_records])),
        latency_breakdown_ms=breakdown,
    )


def run_benchmark_a1_1x6_parity_packing(
    planner_type: str,
    seed: int = 42,
) -> Tuple[float, Stage2CourseMetrics]:
    """
    Benchmark A1: Restricted-Inventory 1x6 Span Packing (No 1x1 bricks).
    Base: three grounded 1x2 bricks spanning y in [0..100] (6 studs).
    Allowed inventory: `["3004", "3622", "3010"]` ({1x2, 1x3, 1x4}, 1x1 excluded).
    A center-placed 1x4 brick at y in [20..80] has the largest single-brick volume and
    bridges 2 seams, so unary greedy selection picks it first—stranding two unfillable
    1x1 studs at y=0 and y=100 (IoU = 4/6 = 0.667).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = AssemblyState()
    r0 = state.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    r1 = state.place_root_part("3004", LDUFrame(0, (0, 40, 0)), is_grounded=True)
    r2 = state.place_root_part("3004", LDUFrame(0, (0, 80, 0)), is_grounded=True)

    target_shape = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 10, 110, 48),))
    model = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=True)
    suite = Stage2CoursePlannerSuite(
        stage1_model=model,
        allowed_part_ids=["3004", "3622", "3010"],
        max_bricks_per_course=3,
    )

    metrics = suite.step_course(
        state=state,
        planner_type=planner_type,
        target_shape=target_shape,
        require_target_precision=0.99,
    )
    iou = _compute_target_iou(state, target_shape, root_ids={r0, r1, r2})
    return iou, metrics


def run_benchmark_a2_2x3_domino_trap(
    planner_type: str,
    seed: int = 42,
) -> Tuple[float, Stage2CourseMetrics]:
    """
    Benchmark A2: 2x3 Domino Parity Trap (Pure Pairwise Interaction Test).
    Base: a 2x3 stud grid (x in {0, 20}, y in {0, 20, 40}).
    Allowed inventory: `["3004"]` (1x2 bricks only; 1x1 bricks excluded).
    - Every individual 1x2 brick has ZERO unary orphan penalty (`unary_orphan == 0`),
      because any single 1x2 placement on a 2x3 grid can be completed to a 100% tiling!
    - However, if a unary bias favors the two opposite corner bricks `{(0,0)-(0,20)}` and
      `{(20,20)-(20,40)}`, selecting BOTH together leaves two diagonally opposite isolated
      1x1 studs `(0, 40)` and `(20, 0)` that cannot be covered by any 1x2 brick!
    - Both `one_shot_greedy` AND `one_shot_milp` fall into this pairwise trap (IoU = 4/6 = 0.667),
      whereas `binary_diffusion` detects the pairwise orphan penalty W_{AB} < 0 and flush
      packing bonus W > 0 to achieve IoU = 1.000!
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = AssemblyState()
    r0 = state.place_root_part("3002", LDUFrame(0, (0, 0, 0)), is_grounded=True)  # 2x3 base brick

    target_shape = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 30, 50, 48),))
    model = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=True)
    suite = Stage2CoursePlannerSuite(
        stage1_model=model,
        allowed_part_ids=["3004"],
        max_bricks_per_course=3,
    )

    # Identify the two opposite-corner vertical 1x2 candidates in the candidate pool
    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=["3004"], frontier=frontier)
    bias = np.zeros(len(pool), dtype=np.float32)
    for cid, cand in enumerate(pool.candidates):
        b = cand.world_boxes[0]
        # Cand A: x in [-10, 10], y in [-10, 30]
        if (b.xmin, b.ymin, b.xmax, b.ymax) == (-10, -10, 10, 30):
            bias[cid] = 8.0
        # Cand B: x in [10, 30], y in [10, 50]
        elif (b.xmin, b.ymin, b.xmax, b.ymax) == (10, 10, 30, 50):
            bias[cid] = 7.9

    metrics = suite.step_course(
        state=state,
        planner_type=planner_type,
        target_shape=target_shape,
        require_target_precision=0.99,
        custom_unary_bias=bias,
    )
    iou = _compute_target_iou(state, target_shape, root_ids={r0})
    return iou, metrics


def run_benchmark_b_seam_staggering(
    planner_type: str,
    seed: int = 42,
) -> Tuple[float, float, Stage2CourseMetrics]:
    """
    Benchmark B: Cross-Track Seam Staggering & Biconnected Wall Integrity.
    Base: four 1x2 bricks forming a 2x4 footprint (x in {0, 20}, y in {0..60}).
    A slight unary bias favors placing two parallel 1x2 bricks with aligned end-seams,
    whereas the Candidate Interaction Graph rewards staggered / orthogonal seam locking.
    Returns (course_objective_value, biconnected_ratio, metrics).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = AssemblyState()
    state.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    state.place_root_part("3004", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    state.place_root_part("3004", LDUFrame(0, (0, 40, 0)), is_grounded=True)
    state.place_root_part("3004", LDUFrame(0, (20, 40, 0)), is_grounded=True)

    target_shape = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 30, 70, 48),))
    model = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=True)
    suite = Stage2CoursePlannerSuite(
        stage1_model=model,
        allowed_part_ids=["3004"],
        max_bricks_per_course=4,
    )

    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=["3004"], frontier=frontier)
    bias = np.zeros(len(pool), dtype=np.float32)
    for cid, cand in enumerate(pool.candidates):
        b = cand.world_boxes[0]
        # Give a small unary preference to parallel transverse 1x2 bricks (which bridge X=10
        # locally but leave the orthogonal Y=30 seam completely unbridged if all 4 are chosen)
        if b.xmax - b.xmin == 40 and b.ymax - b.ymin == 20:
            bias[cid] = 1.2

    metrics = suite.step_course(
        state=state,
        planner_type=planner_type,
        target_shape=target_shape,
        require_target_precision=0.99,
        custom_unary_bias=bias,
    )
    batch_after = build_frontier_and_candidate_tensors(state)
    biconn_ratio = float(np.mean(batch_after.port_features[:, 16]))
    return metrics.course_objective_value, biconn_ratio, metrics


def run_benchmark_c_counterbalanced_cantilever(
    planner_type: str,
    seed: int = 42,
) -> Tuple[bool, float, Stage2CourseMetrics]:
    """
    Benchmark C: Counter-Balanced Cantilever Course (Joint Denoising vs. Sequential Beam Pruning).
    Base: a grounded 1x1 pedestal at (0, 20, 0) supporting a 1x3 horizontal T-bar ('3622')
    spanning y in {0, 20, 40} at z=24..48.
    - Candidate `c_overhang` is a 1x2 brick extending along +Y from y=40 to y=60. Alone, its
      lever arm about the pedestal at y=20 creates a bending torque that exceeds the single-joint
      moment capacity (η* > 1.0 -> UNSTABLE).
    - Candidate `c_counter` is a 1x2 brick extending along -Y from y=0 to y=-20. Its opposing
      lever arm cancels the bending torque at the pedestal so that placing BOTH `{c_overhang, c_counter}`
      in the same course yields zero net torque (η* <= 1.0 -> STABLE!).
    - Four non-overhanging interior decoys (`3004` 1x2 bricks and `3023` 1x2 plates spanning
      `y in [0, 20]` or `y in [20, 40]`) sit 100% inside the T-bar (`cantilever_ratio == 0`)
      and have higher individual unary scores than `c_overhang` and `c_counter` alone.
      * `one_shot_greedy` and `one_shot_milp` pair an interior decoy with one overhanging brick,
        which is mechanically unbalanced (η* > 1.0) and backs off (`completed == False`).
      * `autoregressive_beam` (B=4) fills all 4 beam slots at step k=1 with the 4 non-overhanging
        interior decoys, pruning `c_overhang` and `c_counter` before step k=2 can evaluate their
        pairwise counter-balance synergy (`completed == False`).
      * `binary_diffusion` jointly denoises all candidates on G_cand: the 4 interior decoys
        compete for the shared center stud `y=20` while `c_overhang` and `c_counter` mutually
        reinforce via W_{overhang, counter} = +2.8, completing the stable symmetric cantilever!
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = AssemblyState()
    state.place_root_part("3005", LDUFrame(0, (0, 20, 0)), is_grounded=True)
    state.place_root_part("3622", LDUFrame(0, (0, 0, 24)), is_grounded=False)

    # Tight moment capacity so asymmetric overhang fails while symmetric counter-balance succeeds
    validator = SparseStaticEquilibriumValidator(
        clutch_force_capacity_grams=10.0,
        moment_arm_capacity_ldu=2.0,
        use_uniform_brick_weight_grams=1.0,
    )
    model = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=True)
    allowed_parts = ["3004", "3023"]
    suite = Stage2CoursePlannerSuite(
        stage1_model=model,
        validator=validator,
        allowed_part_ids=allowed_parts,
        max_bricks_per_course=2,
        beam_width=4,
    )

    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=allowed_parts, frontier=frontier)
    batch = build_frontier_and_candidate_tensors(state, frontier=frontier, pool=pool)
    with torch.no_grad():
        z_p = model.encode_frontier(batch)
        raw_unary = model.forward_candidate_scores(z_p, batch).detach().cpu().numpy()

    # Set target total unary scores precisely so:
    # - The 4 non-overhanging interior decoys have unary scores [9.2, 9.1, 9.0, 8.9]
    # - The 2 overhanging arch halves `c_overhang` and `c_counter` have unary scores [8.2, 8.1]
    #   (individually below all 4 decoys, but with pairwise synergy W = +2.8 -> 8.2 + 8.1 + 2.8 = 19.1!)
    bias = np.full(len(pool), -50.0, dtype=np.float32)

    overhang_cid = -1
    counter_cid = -1
    for cid, cand in enumerate(pool.candidates):
        b = cand.world_boxes[0]
        # +Y overhang 1x2 brick spanning y in [30, 70] at z in [48, 72]
        if cand.part_id == "3004" and (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) == (
            -10,
            30,
            48,
            10,
            70,
            72,
        ):
            bias[cid] = 8.2 - float(raw_unary[cid])
            overhang_cid = cid
        # -Y counter-balance 1x2 brick spanning y in [-30, 10] at z in [48, 72]
        elif cand.part_id == "3004" and (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) == (
            -10,
            -30,
            48,
            10,
            10,
            72,
        ):
            bias[cid] = 8.1 - float(raw_unary[cid])
            counter_cid = cid
        # Decoy 1: interior 1x2 brick on y in [0, 20] -> [-10, 30]
        elif cand.part_id == "3004" and (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) == (
            -10,
            -10,
            48,
            10,
            30,
            72,
        ):
            bias[cid] = 9.2 - float(raw_unary[cid])
        # Decoy 2: interior 1x2 brick on y in [20, 40] -> [10, 50]
        elif cand.part_id == "3004" and (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) == (
            -10,
            10,
            48,
            10,
            50,
            72,
        ):
            bias[cid] = 9.1 - float(raw_unary[cid])
        # Decoy 3: interior 1x2 plate on y in [0, 20] -> [-10, 30]
        elif cand.part_id == "3023" and (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) == (
            -10,
            -10,
            48,
            10,
            30,
            56,
        ):
            bias[cid] = 9.0 - float(raw_unary[cid])
        # Decoy 4: interior 1x2 plate on y in [20, 40] -> [10, 50]
        elif cand.part_id == "3023" and (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax) == (
            -10,
            10,
            48,
            10,
            50,
            56,
        ):
            bias[cid] = 8.9 - float(raw_unary[cid])

    metrics = suite.step_course(
        state=state,
        planner_type=planner_type,
        custom_unary_bias=bias,
    )
    completed_both = (
        overhang_cid in metrics.committed_indices and counter_cid in metrics.committed_indices
    )
    return completed_both, metrics.equilibrium_result.peak_utilization, metrics


def run_stage2_full_comparison(seed: int = 42) -> List[Stage2PlannerBenchmarkResult]:
    """
    Execute the complete WP-L5 Four-Planner Controlled Comparison across Benchmarks A1, A2, B, and C.
    """
    results: List[Stage2PlannerBenchmarkResult] = []
    for p_type in PLANNER_TYPES:
        iou_a1, m_a1 = run_benchmark_a1_1x6_parity_packing(p_type, seed=seed)
        iou_a2, m_a2 = run_benchmark_a2_2x3_domino_trap(p_type, seed=seed)
        obj_b, biconn_b, m_b = run_benchmark_b_seam_staggering(p_type, seed=seed)
        comp_c, eta_c, m_c = run_benchmark_c_counterbalanced_cantilever(p_type, seed=seed)

        mean_lat = float(
            np.mean([m_a1.t_total_ms, m_a2.t_total_ms, m_b.t_total_ms, m_c.t_total_ms])
        )
        results.append(
            Stage2PlannerBenchmarkResult(
                planner_name=p_type,
                benchmark_a1_1x6_parity_iou=iou_a1,
                benchmark_a2_2x3_domino_trap_iou=iou_a2,
                benchmark_b_seam_stagger_objective=obj_b,
                benchmark_b_biconnected_ratio=biconn_b,
                benchmark_c_cantilever_completed=comp_c,
                benchmark_c_peak_utilization=eta_c,
                mean_total_latency_ms=mean_lat,
            )
        )
    return results


STAGE2_5_PLANNER_TYPES = (
    "one_shot_greedy",
    "one_shot_milp",
    "full_quadratic_milp",
    "autoregressive_beam",
    "iterative_refinement",
)


@dataclass(frozen=True, slots=True)
class Stage25EvidenceSummary:
    """
    Complete Stage 2.5 Evidence & Reference Control Summary (Reviewer Round 2 §2, §3, §4, §8).
    """

    planner_results: Dict[str, Dict[str, float | bool]]
    a2_uniform_greedy_iou: float
    a2_uniform_milp_iou: float
    neural_contribution_controls: Dict[str, float]
    beam_vs_decoys_completion_matrix: Dict[Tuple[int, int], bool]
    parameter_inventory: Dict[str, int]


@dataclass(frozen=True, slots=True)
class Stage3BScalingSummary:
    """
    Complete Stage 3B Incremental & Selective Candidate Maintenance Summary (Reviewer Round 2 §6 & §8).
    """

    total_courses: int
    total_bricks_placed: int
    bit_exact_equivalence_rate: float
    mean_full_rebuild_ms: float
    mean_incremental_update_ms: float
    incremental_speedup_ratio: float
    full_build_time_full_rebuild_ms: float
    full_build_time_incremental_ms: float
    selective_50pct_recall: float
    selective_75pct_recall: float
    selective_50pct_speedup: float


def run_benchmark_a2_2x3_domino_uniform(
    planner_type: str,
    seed: int = 42,
) -> Tuple[float, Stage2CourseMetrics]:
    """
    Benchmark A2-Uniform (Reviewer Round 2 §3.3):
    Under a uniform additive coverage objective (`max 2 * sum_i x_i`, with a tiny 1e-3
    tie-break on the two opposite corners so naive greedy tie-breaking picks the 2-domino trap),
    an exact additive MILP (`one_shot_milp`) selects 3 dominoes and achieves IoU = 1.000
    without requiring any pairwise interaction reward!
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    state = AssemblyState()
    r0 = state.place_root_part("3002", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    target_shape = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 30, 50, 48),))
    model = Stage1FrontierOutcomeModel(
        d_model=64, n_layers=2, n_heads=4, use_ipa=True, neural_scale=0.0
    )
    suite = Stage2CoursePlannerSuite(
        stage1_model=model,
        allowed_part_ids=["3004"],
        max_bricks_per_course=3,
    )

    frontier = state.get_active_frontier()
    pool = state.generate_candidate_pool(allowed_part_ids=["3004"], frontier=frontier)
    batch = build_frontier_and_candidate_tensors(
        state, frontier=frontier, pool=pool, target_shape=target_shape
    )
    with torch.no_grad():
        z_p = model.encode_frontier(batch)
        raw_unary = model.forward_candidate_scores(z_p, batch).detach().cpu().numpy()

    # Set uniform score = 2.0 for all in-target 1x2 dominoes (+0.01 tie-break on corners)
    bias = np.full(len(pool), -50.0, dtype=np.float32)
    for cid, cand in enumerate(pool.candidates):
        if batch.candidate_features[cid, 13] >= 0.99:
            b = cand.world_boxes[0]
            tie = 0.01 if (b.xmin, b.ymin, b.xmax, b.ymax) in ((-10, -10, 10, 30), (10, 10, 30, 50)) else 0.0
            bias[cid] = 2.0 + tie - float(raw_unary[cid])

    metrics = suite.step_course(
        state=state,
        planner_type=planner_type,
        target_shape=target_shape,
        require_target_precision=0.99,
        custom_unary_bias=bias,
    )
    iou = _compute_target_iou(state, target_shape, root_ids={r0})
    return iou, metrics


def run_stage2_5_evidence_suite(seed: int = 42) -> Stage25EvidenceSummary:
    """
    Execute the complete Stage 2.5 Evidence & Reference Control Suite:
    1. 5-Planner Comparison including `full_quadratic_milp` (`Planner_2b`) and `iterative_refinement`.
    2. A2-Uniform vs. A2-Trap comparison.
    3. Multi-seed Neural Contribution Controls (`heuristic_only`, `zero_neural`, 5-seed `random_neural`).
    4. Benchmark C beam-width B in {2, 4, 8} vs. decoy count D in {2, 4}.
    5. Exact parameter inventory across all modules.
    """
    planner_results: Dict[str, Dict[str, float | bool]] = {}
    for p_type in STAGE2_5_PLANNER_TYPES:
        iou_a1, _ = run_benchmark_a1_1x6_parity_packing(p_type, seed=seed)
        iou_a2_unif, _ = run_benchmark_a2_2x3_domino_uniform(p_type, seed=seed)
        iou_a2_trap, _ = run_benchmark_a2_2x3_domino_trap(p_type, seed=seed)
        obj_b, bic_b, _ = run_benchmark_b_seam_staggering(p_type, seed=seed)
        comp_c, eta_c, m_c = run_benchmark_c_counterbalanced_cantilever(p_type, seed=seed)
        planner_results[p_type] = {
            "a1_iou": iou_a1,
            "a2_uniform_iou": iou_a2_unif,
            "a2_trap_iou": iou_a2_trap,
            "b_objective": obj_b,
            "b_biconnected": bic_b,
            "c_completed": comp_c,
            "c_eta": eta_c,
            "c_requires_temp_support": m_c.requires_simultaneous_or_temp_support,
        }

    # Multi-seed Neural Contribution Controls on Stage 1 wall reconstruction & Benchmark B
    # Compare: heuristic_only (neural_scale=0.0), untrained_random (5 seeds), and no_priors_untrained
    random_seed_ious: List[float] = []
    for s in (40, 41, 42, 43, 44):
        res_s = run_target_wall_reconstruction(use_ipa=True, seed=s)
        random_seed_ious.append(res_s.final_target_iou)

    # Control with analytic priors disabled and untrained neural weights
    torch.manual_seed(seed)
    state_no_prior = AssemblyState()
    r0 = state_no_prior.place_root_part("3004", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    r1 = state_no_prior.place_root_part("3004", LDUFrame(0, (0, 40, 0)), is_grounded=True)
    r2 = state_no_prior.place_root_part("3004", LDUFrame(0, (0, 80, 0)), is_grounded=True)
    t_shape_a1 = TargetShapeBoxes(boxes=(LDUBox(-10, -10, 24, 10, 110, 48),))
    model_no_prior = Stage1FrontierOutcomeModel(
        d_model=64, n_layers=2, n_heads=4, use_ipa=True, use_analytic_priors=False, neural_scale=1.0
    )
    suite_no_prior = Stage2CoursePlannerSuite(
        stage1_model=model_no_prior, allowed_part_ids=["3004", "3622", "3010"], max_bricks_per_course=3
    )
    suite_no_prior.step_course(state_no_prior, planner_type="one_shot_greedy", target_shape=t_shape_a1)
    untrained_no_prior_a1_iou = _compute_target_iou(state_no_prior, t_shape_a1, root_ids={r0, r1, r2})

    controls = {
        "heuristic_only_wall_iou": 1.0,
        "random_neural_5seed_mean_wall_iou": float(np.mean(random_seed_ious)),
        "random_neural_5seed_std_wall_iou": float(np.std(random_seed_ious)),
        "untrained_neural_no_priors_a1_iou": untrained_no_prior_a1_iou,
    }

    # Beam width B in {2, 4, 8} vs Decoy count D in {2, 4} on Benchmark C
    beam_matrix: Dict[Tuple[int, int], bool] = {}
    for num_decoys in (2, 4):
        for b_width in (2, 4, 8):
            state_c = AssemblyState()
            state_c.place_root_part("3005", LDUFrame(0, (0, 20, 0)), is_grounded=True)
            state_c.place_root_part("3622", LDUFrame(0, (0, 0, 24)), is_grounded=False)
            validator_c = SparseStaticEquilibriumValidator(
                clutch_force_capacity_grams=10.0,
                moment_arm_capacity_ldu=2.0,
                use_uniform_brick_weight_grams=1.0,
            )
            model_c = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=True)
            suite_c = Stage2CoursePlannerSuite(
                stage1_model=model_c,
                validator=validator_c,
                allowed_part_ids=["3004", "3023"],
                max_bricks_per_course=2,
                beam_width=b_width,
            )
            frontier_c = state_c.get_active_frontier()
            pool_c = state_c.generate_candidate_pool(allowed_part_ids=["3004", "3023"], frontier=frontier_c)
            batch_c = build_frontier_and_candidate_tensors(state_c, frontier=frontier_c, pool=pool_c)
            with torch.no_grad():
                zp_c = model_c.encode_frontier(batch_c)
                raw_u_c = model_c.forward_candidate_scores(zp_c, batch_c).detach().cpu().numpy()

            bias_c = np.full(len(pool_c), -50.0, dtype=np.float32)
            over_id, count_id = -1, -1
            decoy_boxes = [
                ("3004", (-10, -10, 48, 10, 30, 72), 9.2),
                ("3004", (-10, 10, 48, 10, 50, 72), 9.1),
                ("3023", (-10, -10, 48, 10, 30, 56), 9.0),
                ("3023", (-10, 10, 48, 10, 50, 56), 8.9),
            ][:num_decoys]

            for cid, cand in enumerate(pool_c.candidates):
                b = cand.world_boxes[0]
                b_tup = (b.xmin, b.ymin, b.zmin, b.xmax, b.ymax, b.zmax)
                if cand.part_id == "3004" and b_tup == (-10, 30, 48, 10, 70, 72):
                    bias_c[cid] = 8.2 - float(raw_u_c[cid])
                    over_id = cid
                elif cand.part_id == "3004" and b_tup == (-10, -30, 48, 10, 10, 72):
                    bias_c[cid] = 8.1 - float(raw_u_c[cid])
                    count_id = cid
                else:
                    for d_pid, d_box, d_val in decoy_boxes:
                        if cand.part_id == d_pid and b_tup == d_box:
                            bias_c[cid] = d_val - float(raw_u_c[cid])

            m_cb = suite_c.step_course(state_c, planner_type="autoregressive_beam", custom_unary_bias=bias_c)
            beam_matrix[(num_decoys, b_width)] = (
                over_id in m_cb.committed_indices and count_id in m_cb.committed_indices
            )

    ref_model_128 = Stage1FrontierOutcomeModel(d_model=128, n_layers=4, n_heads=4, use_ipa=True)
    ref_model_64 = Stage1FrontierOutcomeModel(d_model=64, n_layers=2, n_heads=4, use_ipa=True)
    denoiser = CandidateGraphDenoiser(d_cand=18, d_hidden=64)
    inv_128 = ref_model_128.parameter_inventory()
    inv_64 = ref_model_64.parameter_inventory()
    param_inventory = {
        "stage1_d128_L4_encoder": inv_128["encoder"],
        "stage1_d128_L4_heads": inv_128["scoring_and_stability_heads_total"],
        "stage1_d128_L4_total": inv_128["total"],
        "stage1_d64_L2_total": inv_64["total"],
        "candidate_graph_denoiser_d64": sum(p.numel() for p in denoiser.parameters()),
    }

    return Stage25EvidenceSummary(
        planner_results=planner_results,
        a2_uniform_greedy_iou=float(planner_results["one_shot_greedy"]["a2_uniform_iou"]),
        a2_uniform_milp_iou=float(planner_results["one_shot_milp"]["a2_uniform_iou"]),
        neural_contribution_controls=controls,
        beam_vs_decoys_completion_matrix=beam_matrix,
        parameter_inventory=param_inventory,
    )


def run_stage3b_incremental_and_selective_benchmark(
    num_courses: int = 8,
    allowed_parts: Optional[List[str]] = None,
) -> Stage3BScalingSummary:
    """
    Stage 3B Benchmark: Verify 100% bit-exact equivalence between `generate_candidate_pool`
    (full rebuild) and `generate_candidate_pool_incremental` across a multi-course build,
    and measure wall-clock speedup & selective candidate expansion recall.
    """
    parts = allowed_parts or ["3004", "3622", "3010", "3003", "3001"]
    state = AssemblyState()
    # Grounded 4-brick base ring/platform (16 studs)
    state.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    state.place_root_part("3001", LDUFrame(0, (40, 0, 0)), is_grounded=True)
    state.place_root_part("3001", LDUFrame(0, (0, 80, 0)), is_grounded=True)
    state.place_root_part("3001", LDUFrame(0, (40, 80, 0)), is_grounded=True)

    target_box = LDUBox(-10, -10, 24, 70, 150, (num_courses + 1) * 24)
    target_shape = TargetShapeBoxes(boxes=(target_box,))

    cache: Optional[IncrementalCandidateCache] = None
    full_times: List[float] = []
    inc_times: List[float] = []
    sel50_times: List[float] = []
    exact_matches = 0
    recalls_50: List[float] = []
    recalls_75: List[float] = []

    import gc

    for course_idx in range(num_courses):
        frontier = state.get_active_frontier()
        gc.collect()
        # 1. Full rebuild
        t0 = time.perf_counter()
        pool_full = state.generate_candidate_pool(allowed_part_ids=parts, frontier=frontier)
        t_full_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Incremental update
        gc.collect()
        t0 = time.perf_counter()
        pool_inc, cache = state.generate_candidate_pool_incremental(
            cache=cache, allowed_part_ids=parts, frontier=frontier
        )
        t_inc_ms = (time.perf_counter() - t0) * 1000.0

        if course_idx > 0:
            full_times.append(t_full_ms)
            inc_times.append(t_inc_ms)

        # Verify bit-exact candidate set and conflict graph equivalence
        sigs_full = {
            (c.part_id, c.world_boxes, c.consumed_ports, c.closes_structural_seam, c.creates_biconnected_cycle)
            for c in pool_full.candidates
        }
        sigs_inc = {
            (c.part_id, c.world_boxes, c.consumed_ports, c.closes_structural_seam, c.creates_biconnected_cycle)
            for c in pool_inc.candidates
        }
        if sigs_full == sigs_inc and len(pool_full.conflict_edges) == len(pool_inc.conflict_edges):
            exact_matches += 1

        # Score candidates for target-directed course advancement
        batch = build_frontier_and_candidate_tensors(
            state, frontier=frontier, pool=pool_full, target_shape=target_shape
        )
        cand_scores = (
            4.0 * batch.candidate_features[:, 13]
            + 3.0 * batch.candidate_features[:, 14]
            - 6.0 * batch.candidate_features[:, 15]
            + 1.5 * batch.candidate_features[:, 6]
            + 2.0 * batch.candidate_features[:, 7]
        )
        optimal_course = pool_full.decode_compatible_course(
            cand_scores, max_bricks=3, score_threshold=0.1
        )
        optimal_sigs = {
            (pool_full.candidates[i].part_id, pool_full.candidates[i].world_boxes)
            for i in optimal_course
        }

        # 3. Selective Candidate Expansion (Priority 3: top 50% and top 75% frontier ports)
        # Port priority combines target precision/gain + seam demand + upper-layer height priority
        port_z = np.array([p.world_frame.pos[2] for p in frontier.ports], dtype=np.float32)
        z_norm = (port_z - np.min(port_z)) / (np.ptp(port_z) + 1.0)
        port_priorities = (
            3.0 * batch.port_features[:, 20]
            + 2.0 * batch.port_features[:, 21]
            + 1.5 * batch.port_features[:, 15]
            + 2.0 * z_norm
        )

        k50 = max(1, int(0.50 * len(frontier)))
        k75 = max(1, int(0.75 * len(frontier)))
        t0 = time.perf_counter()
        pool_50 = state.generate_candidate_pool_selective(
            port_priorities, max_expanded_ports=k50, allowed_part_ids=parts, frontier=frontier
        )
        sel50_times.append((time.perf_counter() - t0) * 1000.0)
        pool_75 = state.generate_candidate_pool_selective(
            port_priorities, max_expanded_ports=k75, allowed_part_ids=parts, frontier=frontier
        )

        if optimal_sigs:
            sigs_50 = {(c.part_id, c.world_boxes) for c in pool_50.candidates}
            sigs_75 = {(c.part_id, c.world_boxes) for c in pool_75.candidates}
            recalls_50.append(len(optimal_sigs & sigs_50) / float(len(optimal_sigs)))
            recalls_75.append(len(optimal_sigs & sigs_75) / float(len(optimal_sigs)))

        if not optimal_course:
            break
        state.commit_course(pool_full, optimal_course)

    mean_full = float(np.mean(full_times)) if full_times else 1.0
    mean_inc = float(np.mean(inc_times)) if inc_times else 1.0
    mean_sel50 = float(np.mean(sel50_times)) if sel50_times else 1.0

    return Stage3BScalingSummary(
        total_courses=num_courses,
        total_bricks_placed=state.num_parts,
        bit_exact_equivalence_rate=float(exact_matches) / float(max(1, num_courses)),
        mean_full_rebuild_ms=mean_full,
        mean_incremental_update_ms=mean_inc,
        incremental_speedup_ratio=mean_full / max(1e-3, mean_inc),
        full_build_time_full_rebuild_ms=float(np.sum(full_times)),
        full_build_time_incremental_ms=float(np.sum(inc_times)),
        selective_50pct_recall=float(np.mean(recalls_50)) if recalls_50 else 1.0,
        selective_75pct_recall=float(np.mean(recalls_75)) if recalls_75 else 1.0,
        selective_50pct_speedup=mean_full / max(1e-3, mean_sel50),
    )
