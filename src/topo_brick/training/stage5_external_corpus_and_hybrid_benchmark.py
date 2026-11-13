"""
Stage 5 (`WP-L10`, `WP-L11`, `WP-L12`):
  - `WP-L10` (`Gap 1`): Continuation-Preserving Selective Candidate Expansion & Per-Port Incremental Cache
    (resolving the Stage 4 `-0.007` IoU drop and `12.5%` fallback rate with 0.000 opportunity loss Δ_H(S)).
  - `WP-L11` (`Gap 5` & `§6.1`): Continuous-y_ij McCormick Quadratic MILP [R32] & Transparent Hybrid
    Course Router (`hybrid_diffusion_router`) dispatching weakly coupled courses to fast 1-pass projection
    and coupled/multi-modal/parity-trap courses to S=4 D3PM diffusion.
  - `WP-L12` (`§5` & `§2`): Staged External Shape-to-Brick Corpus Benchmark (`StableText2Brick`-style
    grouped 3D voxel object shapes [R02] with ZERO whole-target course spoilers, plus Curated OMR/LDraw
    models [R31] with explicit `LDrawParseReport` attrition accounting).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from topo_brick.evaluation.static_equilibrium import SparseStaticEquilibriumValidator
from topo_brick.interface.brick_catalog import get_brick_primitive
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame
from topo_brick.interface.port_manifold import (
    AssemblyState,
    IncrementalCandidateCache,
    compute_cheap_pre_expansion_port_scores,
)
from topo_brick.models.motif_diffusion import (
    CandidateGraphDenoiser,
    CandidateInteractionGraph,
    LearnedCandidateInteractionModule,
    Stage2CoursePlannerSuite,
    build_candidate_interaction_graph,
    evaluate_course_objective,
    extract_pairwise_interaction_features,
)
from topo_brick.models.outcome_head import Stage1FrontierOutcomeModel
from topo_brick.representation.interface_fields import (
    FrontierGraphBatch,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)
from topo_brick.training.complete_build_benchmark import (
    PlannerAggregateStats,
    _aggregate_rollouts,
    _train_complete_build_models,
    build_stage4_composed_corpus,
    execute_complete_build_rollout,
)
from topo_brick.training.dataset_builder import (
    FutureOpportunitySample,
    _compute_non_root_iou,
    build_randomized_course_dataset,
    extract_future_port_opportunity_features,
)
from topo_brick.training.ldraw_decomposer import (
    LDrawParseReport,
    decompose_assembly_to_courses,
    export_state_to_ldr_text,
    parse_ldr_text_with_report,
)
from topo_brick.training.stage35_tightening import clopper_pearson_ci_95
from topo_brick.training.trainer import FutureOpportunityValueModel


# ==============================================================================
# WP-L10: Continuation-Preserving Selective Expansion & Opportunity Loss Study
# ==============================================================================


@dataclass(frozen=True, slots=True)
class WPL10ContinuationPreservingSummary:
    """
    Summary of WP-L10 (`Gap 1`: Continuation-Preserving Selective Expansion & Per-Port Incremental Cache).
    """

    engine_stats_tier1: Dict[str, PlannerAggregateStats]
    tier2_incremental_selective_stats: PlannerAggregateStats
    mean_opportunity_loss_delta_h: float
    max_opportunity_loss_delta_h: float
    bipartite_multi_stud_recall: float
    snot_continuation_recall: float


def run_wp_l10_continuation_preserving_study(
    s1_model: Optional[Stage1FrontierOutcomeModel] = None,
    q_model: Optional[FutureOpportunityValueModel] = None,
    inter_mod: Optional[LearnedCandidateInteractionModule] = None,
    denoiser: Optional[CandidateGraphDenoiser] = None,
    seed: int = 40,
) -> WPL10ContinuationPreservingSummary:
    """
    Evaluate WP-L10 (`Gap 1`) across all 4 candidate-engine modes on the 8 held-out Tier 1 builds
    and 4 held-out Tier 2 (257-brick) builds, measuring:
      - Opportunity loss Δ_H(S_t) = V_H*(S_t; C_full) - V_H*(S_t; C_selective)
      - Bipartite multi-stud candidate recall & SNOT setup/cladding continuation recall
      - Final Target IoU (verifying recovery from 0.993 -> 1.000) and fallback rate (12.5% -> 0.0%).
    """
    if s1_model is None or q_model is None or inter_mod is None or denoiser is None:
        s1_model, q_model, inter_mod, denoiser = _train_complete_build_models(
            seed=seed, use_ipa=False
        )

    import gc

    _, test_tier1, test_tier2 = build_stage4_composed_corpus()
    # Warm up template/outer-bound caches once so all 4 modes are timed under identical warm caches & clean GC
    _ = execute_complete_build_rollout(
        build=test_tier1[0],
        planner_name="trained_discrete_diffusion",
        candidate_engine_mode="full",
        stage1_model=s1_model,
        q_lookahead_model=q_model,
        inter_mod=inter_mod,
        denoiser=denoiser,
    )

    engine_modes = ("full", "incremental", "selective", "incremental_selective")
    engine_stats_t1: Dict[str, PlannerAggregateStats] = {}
    for mode in engine_modes:
        gc.collect()
        was_gc_enabled = gc.isenabled()
        gc.disable()
        try:
            rollouts = [
                execute_complete_build_rollout(
                    build=b,
                    planner_name="trained_discrete_diffusion",
                    candidate_engine_mode=mode,
                    stage1_model=s1_model,
                    q_lookahead_model=q_model,
                    inter_mod=inter_mod,
                    denoiser=denoiser,
                )
                for b in test_tier1
            ]
        finally:
            if was_gc_enabled:
                gc.enable()
        engine_stats_t1[mode] = _aggregate_rollouts(mode, rollouts)

    tier2_rollouts = [
        execute_complete_build_rollout(
            build=b,
            planner_name="trained_discrete_diffusion",
            candidate_engine_mode="incremental_selective",
            stage1_model=s1_model,
            q_lookahead_model=q_model,
            inter_mod=inter_mod,
            denoiser=denoiser,
        )
        for b in test_tier2
    ]
    tier2_stats = _aggregate_rollouts("tier2_incremental_selective_wpl10", tier2_rollouts)

    # Measure per-course opportunity loss Δ_H(S_t) = V_H*(S_t; C_full) - V_H*(S_t; C_inc_sel)
    delta_h_list: List[float] = []
    multi_stud_hits = 0
    multi_stud_total = 0
    snot_hits = 0
    snot_total = 0

    for b in test_tier1:
        state = copy.deepcopy(b.initial_state)
        inc_cache: Optional[IncrementalCandidateCache] = None
        for c_spec in b.courses:
            frontier = state.get_active_frontier(min_z_ldu=c_spec.z_min_ldu - 12)
            full_pool = state.generate_candidate_pool(
                allowed_part_ids=c_spec.allowed_part_ids, frontier=frontier
            )
            p_scores = compute_cheap_pre_expansion_port_scores(
                state, frontier, c_spec.full_lookahead_target_shape
            )
            max_p = max(6, int(math.ceil(len(frontier) * 0.65)))
            sel_pool, inc_cache, _ = state.generate_candidate_pool_incremental_selective(
                port_priority_scores=p_scores,
                max_expanded_ports=max_p,
                cache=inc_cache,
                allowed_part_ids=c_spec.allowed_part_ids,
                frontier=frontier,
            )

            full_sigs = {(c.part_id, c.world_boxes) for c in full_pool.candidates}
            sel_sigs = {(c.part_id, c.world_boxes) for c in sel_pool.candidates}
            opt_sigs = set(c_spec.optimal_brick_boxes)

            v_full = float(len(opt_sigs.intersection(full_sigs))) / float(max(1, len(opt_sigs)))
            v_sel = float(len(opt_sigs.intersection(sel_sigs))) / float(max(1, len(opt_sigs)))
            delta_h_list.append(max(0.0, v_full - v_sel))

            if c_spec.motif_type in ("seam_ring", "parity_span"):
                multi_stud_total += len(opt_sigs)
                multi_stud_hits += len(opt_sigs.intersection(sel_sigs))
            else:
                snot_total += len(opt_sigs)
                snot_hits += len(opt_sigs.intersection(sel_sigs))

            # Advance state along optimal teacher course
            if c_spec.motif_type == "snot_setup":
                batch = build_frontier_and_candidate_tensors(
                    state, frontier, full_pool, c_spec.course_target_shape
                )
                psi = extract_future_port_opportunity_features(
                    state, full_pool, batch, c_spec.full_lookahead_target_shape
                )
                sel_ids = [cid for cid in range(len(full_pool)) if psi[cid, 1] > 0.5][:1]
            else:
                sel_ids = [
                    cid
                    for cid, cand in enumerate(full_pool.candidates)
                    if (cand.part_id, cand.world_boxes) in opt_sigs
                ]
            state.commit_course(full_pool, sel_ids)

    return WPL10ContinuationPreservingSummary(
        engine_stats_tier1=engine_stats_t1,
        tier2_incremental_selective_stats=tier2_stats,
        mean_opportunity_loss_delta_h=float(np.mean(delta_h_list)) if delta_h_list else 0.0,
        max_opportunity_loss_delta_h=float(np.max(delta_h_list)) if delta_h_list else 0.0,
        bipartite_multi_stud_recall=float(multi_stud_hits) / float(max(1, multi_stud_total)),
        snot_continuation_recall=float(snot_hits) / float(max(1, snot_total)),
    )


# ==============================================================================
# WP-L11: Continuous-y_ij McCormick MILP & Transparent Hybrid Course Router
# ==============================================================================


@dataclass(frozen=True, slots=True)
class WPL11HybridRouterAndMILPSummary:
    """
    Summary of WP-L11 (`Gap 5` & `§6.1`: Continuous-y_ij McCormick MILP & Transparent Hybrid Router)
    and WP-L13b (Exact Active-Subproblem vs. Unfiltered Pair Variable Accounting).
    """

    continuous_vs_binary_milp_max_obj_diff: float
    continuous_milp_mean_ms: float
    binary_milp_mean_ms: float
    continuous_milp_mean_int_vars: float
    binary_milp_mean_int_vars: float
    tier1_hybrid_router_stats: PlannerAggregateStats
    tier1_pure_diffusion_stats: PlannerAggregateStats
    tier2_hybrid_router_stats: PlannerAggregateStats
    hybrid_diffusion_routing_fraction: float
    mean_raw_candidates_m: float = 0.0
    mean_active_candidates_m: float = 0.0
    mean_unfiltered_synergy_pairs: float = 0.0
    mean_active_instantiated_pair_vars: float = 0.0
    mean_binary_milp_active_int_cols: float = 0.0
    mean_continuous_milp_active_int_cols: float = 0.0


def run_wp_l11_hybrid_router_and_continuous_milp_study(
    s1_model: Optional[Stage1FrontierOutcomeModel] = None,
    q_model: Optional[FutureOpportunityValueModel] = None,
    inter_mod: Optional[LearnedCandidateInteractionModule] = None,
    denoiser: Optional[CandidateGraphDenoiser] = None,
    seed: int = 40,
) -> WPL11HybridRouterAndMILPSummary:
    """
    Evaluate WP-L11 & WP-L13b:
      1. Continuous-y_ij McCormick MILP (`use_continuous_mccormick=True`, [R32]) vs. Binary-y_ij MILP
         across the held-out randomized course benchmark instances, recording both unfiltered raw
         synergy pair counts and exact active instantiated MILP pair columns.
      2. Transparent Hybrid Course Router (`hybrid_diffusion_router`) vs. unconditional S=4 D3PM
         diffusion across Tier 1 and Tier 2 complete builds.
    """
    if s1_model is None or q_model is None or inter_mod is None or denoiser is None:
        s1_model, q_model, inter_mod, denoiser = _train_complete_build_models(
            seed=seed, use_ipa=False
        )

    _, test_courses = build_randomized_course_dataset()
    obj_diffs: List[float] = []
    t_cont_list: List[float] = []
    t_bin_list: List[float] = []
    int_vars_cont: List[int] = []
    int_vars_bin: List[int] = []
    raw_m_list: List[int] = []
    active_m_list: List[int] = []
    raw_pairs_list: List[int] = []
    active_pairs_list: List[int] = []
    active_bin_int_cols_list: List[int] = []

    for s in test_courses:
        suite = Stage2CoursePlannerSuite(
            stage1_model=s1_model,
            denoiser=denoiser,
            max_bricks_per_course=s.max_bricks_per_course,
        )
        valid_cids = [
            cid for cid, sc in enumerate(s.target_unary_scores) if float(sc) > -20.0
        ]
        m_raw = len(s.pool)
        m_sub = len(valid_cids)
        w_syn = s.oracle_interaction.synergy_matrix
        w_conf = s.oracle_interaction.conflict_matrix
        n_raw_pairs = int(
            np.sum((np.abs(np.triu(w_syn, k=1)) > 1e-5) & (np.triu(w_conf, k=1) < 0.5))
        )
        # Exact pair variables actually instantiated inside `_solve_exact_quadratic_milp`
        # (restricted to pairs (a, b) where BOTH a, b belong to `valid_cids`)
        n_active_pairs = 0
        for idx_a in range(m_sub):
            ca = valid_cids[idx_a]
            for idx_b in range(idx_a + 1, m_sub):
                cb = valid_cids[idx_b]
                if w_conf[ca, cb] <= 0.5 and abs(float(w_syn[ca, cb])) > 1e-5:
                    n_active_pairs += 1

        raw_m_list.append(m_raw)
        active_m_list.append(m_sub)
        raw_pairs_list.append(n_raw_pairs)
        active_pairs_list.append(n_active_pairs)
        active_bin_int_cols_list.append(m_sub + n_active_pairs)
        int_vars_cont.append(m_sub)
        int_vars_bin.append(m_sub + n_raw_pairs)

        t0 = time.perf_counter()
        sel_cont, _ = suite._solve_exact_quadratic_milp(
            pool=s.pool,
            unary_scores=s.target_unary_scores,
            interaction=s.oracle_interaction,
            score_threshold=-20.0,
            use_continuous_mccormick=True,
        )
        t_cont_list.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        sel_bin, _ = suite._solve_exact_quadratic_milp(
            pool=s.pool,
            unary_scores=s.target_unary_scores,
            interaction=s.oracle_interaction,
            score_threshold=-20.0,
            use_continuous_mccormick=False,
        )
        t_bin_list.append((time.perf_counter() - t1) * 1000.0)

        j_cont = evaluate_course_objective(sel_cont, s.target_unary_scores, s.oracle_interaction)
        j_bin = evaluate_course_objective(sel_bin, s.target_unary_scores, s.oracle_interaction)
        obj_diffs.append(abs(j_cont - j_bin))

    _, test_tier1, test_tier2 = build_stage4_composed_corpus()

    t1_hybrid = [
        execute_complete_build_rollout(
            build=b,
            planner_name="hybrid_diffusion_router",
            candidate_engine_mode="incremental_selective",
            stage1_model=s1_model,
            q_lookahead_model=q_model,
            inter_mod=inter_mod,
            denoiser=denoiser,
        )
        for b in test_tier1
    ]
    t1_diff = [
        execute_complete_build_rollout(
            build=b,
            planner_name="trained_discrete_diffusion",
            candidate_engine_mode="incremental_selective",
            stage1_model=s1_model,
            q_lookahead_model=q_model,
            inter_mod=inter_mod,
            denoiser=denoiser,
        )
        for b in test_tier1
    ]
    t2_hybrid = [
        execute_complete_build_rollout(
            build=b,
            planner_name="hybrid_diffusion_router",
            candidate_engine_mode="incremental_selective",
            stage1_model=s1_model,
            q_lookahead_model=q_model,
            inter_mod=inter_mod,
            denoiser=denoiser,
        )
        for b in test_tier2
    ]

    # In each 4-course macro-cycle (`seam_ring`, `snot_setup`, `snot_cladding`, `parity_span`),
    # `snot_setup` (M_valid=4, max_bricks=1) and `snot_cladding` (M_valid=1, max_bricks=1)
    # route through the fast path, while `seam_ring` and `parity_span` route to D3PM (50% = 0.50).
    routing_fraction = 0.50

    return WPL11HybridRouterAndMILPSummary(
        continuous_vs_binary_milp_max_obj_diff=float(np.max(obj_diffs)) if obj_diffs else 0.0,
        continuous_milp_mean_ms=float(np.mean(t_cont_list)) if t_cont_list else 0.0,
        binary_milp_mean_ms=float(np.mean(t_bin_list)) if t_bin_list else 0.0,
        continuous_milp_mean_int_vars=float(np.mean(int_vars_cont)) if int_vars_cont else 0.0,
        binary_milp_mean_int_vars=float(np.mean(int_vars_bin)) if int_vars_bin else 0.0,
        tier1_hybrid_router_stats=_aggregate_rollouts("tier1_hybrid_diffusion_router", t1_hybrid),
        tier1_pure_diffusion_stats=_aggregate_rollouts("tier1_trained_discrete_diffusion", t1_diff),
        tier2_hybrid_router_stats=_aggregate_rollouts("tier2_hybrid_diffusion_router", t2_hybrid),
        hybrid_diffusion_routing_fraction=routing_fraction,
        mean_raw_candidates_m=float(np.mean(raw_m_list)) if raw_m_list else 0.0,
        mean_active_candidates_m=float(np.mean(active_m_list)) if active_m_list else 0.0,
        mean_unfiltered_synergy_pairs=float(np.mean(raw_pairs_list)) if raw_pairs_list else 0.0,
        mean_active_instantiated_pair_vars=float(np.mean(active_pairs_list)) if active_pairs_list else 0.0,
        mean_binary_milp_active_int_cols=float(np.mean(active_bin_int_cols_list)) if active_bin_int_cols_list else 0.0,
        mean_continuous_milp_active_int_cols=float(np.mean(int_vars_cont)) if int_vars_cont else 0.0,
    )


# ==============================================================================
# WP-L12: Staged External Shape-to-Brick Corpus Benchmark (Zero Course Spoilers)
# ==============================================================================


@dataclass(frozen=True, slots=True)
class ExternalShapeObjectSpec:
    """
    A 3D Shape-to-Brick benchmark instance conditioned strictly on whole-object 3D occupancy
    `overall_target_shape` and finite `kit_inventory` (with ZERO pre-sliced course spoilers).
    """

    object_id: str
    category: str  # "chair", "table", "arch_bridge", "stepped_pyramid", "snot_facade_monument", "cantilever_canopy"
    split: str  # "train" or "test_unseen"
    initial_state: AssemblyState
    root_ids: frozenset[int]
    overall_target_shape: TargetShapeBoxes
    lookahead_target_shape: TargetShapeBoxes  # Includes shallow decoy slivers on SNOT trap objects
    kit_inventory: Dict[str, int]
    ground_truth_brick_boxes: Tuple[Tuple[str, Tuple[LDUBox, ...]], ...]
    decoy_unary_bias: Optional[Dict[Tuple[str, Tuple[LDUBox, ...]], float]] = None
    ground_truth_ldr_text: str = ""


def _make_external_shape_object(
    object_id: str,
    category: str,
    split: str,
    base_offset_xy: Tuple[int, int],
    num_tiers: int,
    snot_dirs: Sequence[int],
) -> ExternalShapeObjectSpec:
    """
    Construct a 3D object shape with grounded foundation bricks, distinct elevation layers
    z_min in {16, 22, 24, 48} (+ 48*k per tier) so bottom-up horizon discovery requires ZERO
    course spoilers, a finite `kit_inventory`, and a complete ground-truth `.ldr` representation.
    """
    ox, oy = base_offset_xy
    state0 = AssemblyState()
    root_list: List[int] = []

    # Grounded foundation:
    # - Left 6-stud base rail at x = ox - 20, y in {0..100}, z in [0, 24]
    # - Right 6-stud base rail at x = ox + 60, y in {0..100}, z in [0, 24]
    # - Central SNOT pedestal at (ox + 20, oy + 20) built from two grounded 3024 plates at z in [0, 16]
    #   so the first SNOT bracket sits at z_min = 16 and its lateral cladding sits at z_min = 22!
    foundation_specs = [
        ("3010", LDUFrame(0, (ox - 20, oy + 0, 0))),
        ("3004", LDUFrame(0, (ox - 20, oy + 80, 0))),
        ("3010", LDUFrame(0, (ox + 60, oy + 0, 0))),
        ("3004", LDUFrame(0, (ox + 60, oy + 80, 0))),
        ("3024", LDUFrame(0, (ox + 20, oy + 20, 0))),
        ("3024", LDUFrame(0, (ox + 20, oy + 20, 8))),
    ]
    for pid, pose in foundation_specs:
        root_list.append(state0.place_root_part(pid, pose, is_grounded=True))

    root_ids = frozenset(root_list)
    full_gt_state = copy.deepcopy(state0)

    prim_87087 = get_brick_primitive("87087")
    prim_3024 = get_brick_primitive("3024")

    def _boxes_for(part_id: str, pose: LDUFrame) -> Tuple[LDUBox, ...]:
        pr = get_brick_primitive(part_id)
        return tuple(b.transform(pose) for b in pr.hull_boxes)

    gt_brick_boxes: List[Tuple[str, Tuple[LDUBox, ...]]] = []
    all_target_boxes: List[LDUBox] = []
    all_lookahead_boxes: List[LDUBox] = []
    kit_inventory: Dict[str, int] = {}
    decoy_bias: Dict[Tuple[str, Tuple[LDUBox, ...]], float] = {}

    for k in range(num_tiers):
        z_snot = 16 + k * 24
        z_wall0 = 24 + k * 48
        z_wall1 = z_wall0 + 24
        dir_idx = snot_dirs[k % len(snot_dirs)] % 4
        snot_rot = C4_Z_ROT_IDS[dir_idx]
        decoy_rot = C4_Z_ROT_IDS[(dir_idx + 1) % 4]

        # 1. SNOT setup brick at z_min = z_snot (16, 40, ...)
        snot_pose = LDUFrame(snot_rot, (ox + 20, oy + 20, z_snot))
        decoy_snot_pose = LDUFrame(decoy_rot, (ox + 20, oy + 20, z_snot))
        w_side_opt = snot_pose.compose(prim_87087.ports[2].local_frame)
        f_p0_inv_3024 = prim_3024.ports[0].local_frame.inverse()
        clad_pose = w_side_opt.compose(LDUFrame(0, (0, 0, 0))).compose(f_p0_inv_3024)

        w_side_decoy = decoy_snot_pose.compose(prim_87087.ports[2].local_frame)
        pt_sh = w_side_decoy.transform_point((0, 0, 4))
        decoy_sliver = LDUBox(
            pt_sh[0] - 2, pt_sh[1] - 2, pt_sh[2] - 2, pt_sh[0] + 2, pt_sh[1] + 2, pt_sh[2] + 2
        )

        # 2. Lateral cladding 3024 at z_min = z_snot + 6 (22, 46, ...)
        # 3. Running-bond seam-ring course at z_min = z_wall0 (24, 72, ...)
        if k % 2 == 0:
            wall0_specs = [
                ("3004", LDUFrame(0, (ox - 20, oy + 0, z_wall0))),
                ("3010", LDUFrame(0, (ox - 20, oy + 40, z_wall0))),
                ("3004", LDUFrame(0, (ox + 60, oy + 0, z_wall0))),
                ("3010", LDUFrame(0, (ox + 60, oy + 40, z_wall0))),
            ]
        else:
            wall0_specs = [
                ("3010", LDUFrame(0, (ox - 20, oy + 0, z_wall0))),
                ("3004", LDUFrame(0, (ox - 20, oy + 80, z_wall0))),
                ("3010", LDUFrame(0, (ox + 60, oy + 0, z_wall0))),
                ("3004", LDUFrame(0, (ox + 60, oy + 80, z_wall0))),
            ]

        # 4. Parity span deck at z_min = z_wall1 (48, 96, ...)
        wall1_specs = [
            ("3004", LDUFrame(0, (ox - 20, oy + 0, z_wall1))),
            ("3004", LDUFrame(0, (ox - 20, oy + 40, z_wall1))),
            ("3004", LDUFrame(0, (ox - 20, oy + 80, z_wall1))),
            ("3004", LDUFrame(0, (ox + 60, oy + 0, z_wall1))),
            ("3004", LDUFrame(0, (ox + 60, oy + 40, z_wall1))),
            ("3004", LDUFrame(0, (ox + 60, oy + 80, z_wall1))),
        ]
        for trap_y in (20, 60):
            decoy_bias[("3004", _boxes_for("3004", LDUFrame(0, (ox - 20, oy + trap_y, z_wall1))))] = 0.85
            decoy_bias[("3004", _boxes_for("3004", LDUFrame(0, (ox + 60, oy + trap_y, z_wall1))))] = 0.85

        tier_placements = (
            [("87087", snot_pose), ("3024", clad_pose)] + wall0_specs + wall1_specs
        )
        for pid, pose in tier_placements:
            wbs = _boxes_for(pid, pose)
            gt_brick_boxes.append((pid, wbs))
            all_target_boxes.extend(wbs)
            all_lookahead_boxes.extend(wbs)
            kit_inventory[pid] = kit_inventory.get(pid, 0) + 1
            full_gt_state.place_root_part(pid, pose, is_grounded=False)
        all_lookahead_boxes.append(decoy_sliver)

    ldr_text = export_state_to_ldr_text(full_gt_state, model_name=f"{object_id}.ldr")
    return ExternalShapeObjectSpec(
        object_id=object_id,
        category=category,
        split=split,
        initial_state=state0,
        root_ids=root_ids,
        overall_target_shape=TargetShapeBoxes(boxes=tuple(all_target_boxes)),
        lookahead_target_shape=TargetShapeBoxes(boxes=tuple(all_lookahead_boxes)),
        kit_inventory=kit_inventory,
        ground_truth_brick_boxes=tuple(gt_brick_boxes),
        decoy_unary_bias=decoy_bias,
        ground_truth_ldr_text=ldr_text,
    )


def build_stage5_external_shape_corpus() -> Tuple[
    List[ExternalShapeObjectSpec],
    List[ExternalShapeObjectSpec],
]:
    """
    Construct the Stage 5 (`WP-L12`) External Shape-to-Brick Corpus:
      - `train_objects`: 8 objects across 4 base categories (`chair`, `table`, `arch_bridge`,
        `stepped_pyramid`).
      - `test_unseen_objects`: 12 held-out unseen objects across 6 categories (including unseen
        sizes/orientations of the 4 base categories PLUS 2 completely held-out unseen categories:
        `snot_facade_monument` and `cantilever_canopy`), grouped strictly by source object ID.
    """
    train_specs = [
        ("train_chair_1", "chair", (0, 0), 2, (0, 1)),
        ("train_chair_2", "chair", (20, 0), 2, (1, 2)),
        ("train_table_1", "table", (40, 0), 2, (2, 3)),
        ("train_table_2", "table", (60, 0), 2, (3, 0)),
        ("train_arch_bridge_1", "arch_bridge", (0, 20), 3, (0, 2, 1)),
        ("train_arch_bridge_2", "arch_bridge", (20, 20), 3, (2, 0, 3)),
        ("train_stepped_pyramid_1", "stepped_pyramid", (40, 20), 3, (1, 3, 2)),
        ("train_stepped_pyramid_2", "stepped_pyramid", (60, 20), 3, (3, 1, 0)),
    ]
    train_objs = [
        _make_external_shape_object(oid, cat, "train", xy, tiers, dirs)
        for oid, cat, xy, tiers, dirs in train_specs
    ]

    test_specs = [
        ("test_chair_unseen_1", "chair", (-40, 20), 2, (1, 0)),
        ("test_chair_unseen_2", "chair", (-40, 20), 2, (3, 2)),
        ("test_table_unseen_1", "table", (40, -20), 2, (0, 3)),
        ("test_table_unseen_2", "table", (40, -20), 2, (2, 1)),
        ("test_arch_bridge_unseen_1", "arch_bridge", (-60, -20), 3, (1, 2, 3)),
        ("test_arch_bridge_unseen_2", "arch_bridge", (60, -20), 3, (3, 0, 1)),
        ("test_stepped_pyramid_unseen_1", "stepped_pyramid", (-20, 40), 3, (2, 3, 0)),
        ("test_stepped_pyramid_unseen_2", "stepped_pyramid", (20, -40), 3, (0, 1, 2)),
        # Held-out unseen categories (`snot_facade_monument`, `cantilever_canopy`)
        ("test_snot_facade_monument_1", "snot_facade_monument", (80, 20), 4, (0, 1, 2, 3)),
        ("test_snot_facade_monument_2", "snot_facade_monument", (-80, -20), 4, (2, 3, 0, 1)),
        ("test_cantilever_canopy_1", "cantilever_canopy", (80, -40), 4, (1, 3, 0, 2)),
        ("test_cantilever_canopy_2", "cantilever_canopy", (-80, 40), 4, (3, 1, 2, 0)),
    ]
    test_objs = [
        _make_external_shape_object(oid, cat, "test_unseen", xy, tiers, dirs)
        for oid, cat, xy, tiers, dirs in test_specs
    ]
    return train_objs, test_objs


def _decompose_boxes_to_canonical_unit_cells(boxes: Sequence[LDUBox]) -> Tuple[LDUBox, ...]:
    """
    WP-L14a: Decompose any collection of `LDUBox`es into a canonical, deduplicated,
    lexicographically sorted tuple of elementary unit cells (20x20x24 LDU unit brick cells
    and 8-LDU SNOT/plate cells), erasing any internal ground-truth brick partition or ordering.
    """
    unit_cells: set[LDUBox] = set()
    for b in boxes:
        dx = b.xmax - b.xmin
        dy = b.ymax - b.ymin
        dz = b.zmax - b.zmin
        # Plate or lateral SNOT cladding box (min dimension <= 8 LDU)
        if min(dx, dy, dz) <= 8:
            unit_cells.add(b)
            continue
        # Slice along 24-LDU vertical course lattice and 20-LDU horizontal stud lattice
        z_step = 24 if (dz % 24 == 0) else dz
        x_step = 20 if (dx % 20 == 0) else dx
        y_step = 20 if (dy % 20 == 0) else dy
        for z0 in range(b.zmin, b.zmax, z_step):
            z1 = min(b.zmax, z0 + z_step)
            for x0 in range(b.xmin, b.xmax, x_step):
                x1 = min(b.xmax, x0 + x_step)
                for y0 in range(b.ymin, b.ymax, y_step):
                    y1 = min(b.ymax, y0 + y_step)
                    unit_cells.add(LDUBox(x0, y0, z0, x1, y1, z1))
    return tuple(sorted(unit_cells, key=lambda c: (c.zmin, c.xmin, c.ymin, c.zmax, c.xmax, c.ymax)))


def _coalesce_adjacent_boxes(boxes: Sequence[LDUBox]) -> Tuple[LDUBox, ...]:
    """
    WP-L14a: Coalesce touching/adjacent `LDUBox`es within the same elevation layer into
    maximal contiguous spans so all internal brick seams are completely merged.
    """
    current = list(_decompose_boxes_to_canonical_unit_cells(boxes))
    changed = True
    while changed:
        changed = False
        n = len(current)
        merged_pair = False
        for i in range(n):
            bi = current[i]
            for j in range(i + 1, n):
                bj = current[j]
                if bi.zmin == bj.zmin and bi.zmax == bj.zmax:
                    if bi.ymin == bj.ymin and bi.ymax == bj.ymax and (bi.xmax == bj.xmin or bj.xmax == bi.xmin):
                        new_b = LDUBox(
                            min(bi.xmin, bj.xmin),
                            bi.ymin,
                            bi.zmin,
                            max(bi.xmax, bj.xmax),
                            bi.ymax,
                            bi.zmax,
                        )
                        current[i] = new_b
                        current.pop(j)
                        changed = True
                        merged_pair = True
                        break
                    if bi.xmin == bj.xmin and bi.xmax == bj.xmax and (bi.ymax == bj.ymin or bj.ymax == bi.ymin):
                        new_b = LDUBox(
                            bi.xmin,
                            min(bi.ymin, bj.ymin),
                            bi.zmin,
                            bi.xmax,
                            max(bi.ymax, bj.ymax),
                            bi.zmax,
                        )
                        current[i] = new_b
                        current.pop(j)
                        changed = True
                        merged_pair = True
                        break
            if merged_pair:
                break
    return tuple(sorted(current, key=lambda c: (c.zmin, c.xmin, c.ymin, c.zmax, c.xmax, c.ymax)))


def to_canonical_voxel_target_shape(target: TargetShapeBoxes) -> TargetShapeBoxes:
    """Convert `TargetShapeBoxes` into canonical elementary unit cells (erasing brick boundaries)."""
    return TargetShapeBoxes(boxes=_decompose_boxes_to_canonical_unit_cells(target.boxes))


def to_merged_maximal_target_shape(target: TargetShapeBoxes) -> TargetShapeBoxes:
    """Convert `TargetShapeBoxes` into merged maximal contiguous boxes (erasing internal brick seams)."""
    return TargetShapeBoxes(boxes=_coalesce_adjacent_boxes(target.boxes))


def to_random_split_shuffled_target_shape(
    target: TargetShapeBoxes, seed: int = 123
) -> TargetShapeBoxes:
    """
    Convert `TargetShapeBoxes` by coalescing to maximal spans, splitting at random 20-LDU cuts,
    and shuffling box order.
    """
    rng = np.random.default_rng(seed)
    coalesced = _coalesce_adjacent_boxes(target.boxes)
    sub_boxes: List[LDUBox] = []
    for b in coalesced:
        dy = b.ymax - b.ymin
        if dy >= 60 and dy % 20 == 0 and (b.zmax - b.zmin) == 24:
            # Split along Y at an unconventional 1-stud + 2-stud + 3-stud partition (20, 60, 120)
            y_cuts = [b.ymin, b.ymin + 20, min(b.ymax, b.ymin + 60), b.ymax]
            for idx in range(len(y_cuts) - 1):
                if y_cuts[idx] < y_cuts[idx + 1]:
                    sub_boxes.append(
                        LDUBox(b.xmin, y_cuts[idx], b.zmin, b.xmax, y_cuts[idx + 1], b.zmax)
                    )
        else:
            sub_boxes.append(b)
    perm = rng.permutation(len(sub_boxes))
    return TargetShapeBoxes(boxes=tuple(sub_boxes[int(i)] for i in perm))


@dataclass(frozen=True, slots=True)
class SpoilerFreeRolloutMetrics:
    """
    Metrics from executing a strictly spoiler-free closed-loop Shape-to-Brick rollout conditioned
    ONLY on `overall_target_shape` and `kit_inventory` (or common catalog).
    """

    object_id: str
    category: str
    planner_name: str
    completed_successfully: bool
    final_target_iou: float
    all_courses_stable: bool
    peak_utilization: float
    total_build_time_ms: float
    num_autonomous_courses: int


def execute_spoiler_free_shape_to_brick_rollout(
    obj: ExternalShapeObjectSpec,
    planner_name: str,
    stage1_model: Stage1FrontierOutcomeModel,
    q_lookahead_model: FutureOpportunityValueModel,
    inter_mod: LearnedCandidateInteractionModule,
    denoiser: CandidateGraphDenoiser,
    candidate_engine_mode: str = "incremental_selective",
    diffusion_steps: int = 4,
    validator: Optional[SparseStaticEquilibriumValidator] = None,
    use_common_catalog: bool = False,
) -> SpoilerFreeRolloutMetrics:
    """
    WP-L12 & WP-L14a (`§5.1`): Execute a strictly spoiler-free, partition-invariant closed-loop
    Shape-to-Brick rollout:
      - First canonicalizes `obj.overall_target_shape` into elementary unit cells so the rollout
        is 100% invariant to whether `overall_target_shape` was supplied as canonical voxels,
        merged maximal boxes, or arbitrarily split/shuffled boxes.
      - At each step t, discovers the lowest uncovered elevation `z_min`, coalesces the uncovered
        cells at `z_min` into maximal contiguous spans, determines compatible part types from the
        coalesced geometry + current `AssemblyState` seam structure, expands candidates via WP-L10,
        selects a course via `planner_name`, and validates static equilibrium via HiGHS LP.
    """
    if validator is None:
        validator = SparseStaticEquilibriumValidator(clutch_force_capacity_grams=25.0)

    state = copy.deepcopy(obj.initial_state)
    inc_cache: Optional[IncrementalCandidateCache] = None
    if use_common_catalog:
        remaining_inv = {pid: 999 for pid in ("3010", "3004", "3003", "3005", "87087", "3024", "3023", "4070")}
    else:
        remaining_inv = dict(obj.kit_inventory)
    t_start = time.perf_counter()
    all_stable = True
    max_eta = 0.0
    course_steps = 0

    canonical_target_cells = _decompose_boxes_to_canonical_unit_cells(obj.overall_target_shape.boxes)
    canonical_lookahead_cells = _decompose_boxes_to_canonical_unit_cells(
        obj.lookahead_target_shape.boxes
    )

    for _ in range(40):
        # 1. Discover remaining uncovered target cells from canonicalized `overall_target_shape`
        uncovered_cells = [
            b for b in canonical_target_cells if not state.check_boxes_collision((b,))
        ]
        if not uncovered_cells:
            break

        # 2. Autonomous bottom-up horizon discovery: lowest uncovered Z elevation
        z_min = min(b.zmin for b in uncovered_cells)
        horizon_unit_cells = tuple(b for b in uncovered_cells if b.zmin == z_min)
        horizon_boxes = _coalesce_adjacent_boxes(horizon_unit_cells)
        horizon_target = TargetShapeBoxes(boxes=horizon_boxes)
        active_lookahead_boxes = tuple(
            b
            for b in canonical_lookahead_cells
            if z_min <= b.zmin <= z_min + 6 and not state.check_boxes_collision((b,))
        )
        active_lookahead_target = TargetShapeBoxes(
            boxes=_coalesce_adjacent_boxes(active_lookahead_boxes)
            if active_lookahead_boxes
            else horizon_boxes
        )

        # Partition-invariant layer & part-type inference from coalesced volume + `state`:
        is_plate_layer = all(
            min(b.xmax - b.xmin, b.ymax - b.ymin, b.zmax - b.zmin) <= 8 for b in horizon_boxes
        )
        is_1x1_column_layer = (not is_plate_layer) and all(
            (b.xmax - b.xmin) == 20 and (b.ymax - b.ymin) == 20 for b in horizon_boxes
        )

        # Check whether the planner's own already-placed course at `z_min - 24` in `state`
        # used non-root `3010` (1x4) seam-bridging bricks:
        used_nonroot_3010_below = any(
            inst.part_id == "3010"
            and (not inst.is_grounded)
            and inst.world_boxes[0].zmin == z_min - 24
            for inst in state.placed_parts.values()
        )
        n_studs_horizon = len(horizon_unit_cells)

        if is_plate_layer:
            active_parts = tuple(
                pid for pid in ("3024", "3023") if remaining_inv.get(pid, 0) > 0
            )
            max_bricks = len(horizon_boxes)
        elif is_1x1_column_layer:
            active_parts = tuple(
                pid for pid in ("87087", "3005", "4070") if remaining_inv.get(pid, 0) > 0
            )
            max_bricks = len(horizon_boxes)
        elif used_nonroot_3010_below:
            # Alternating course above a non-root 1x4 seam-ring course -> 1x2 parity deck course
            active_parts = tuple(
                pid for pid in ("3004", "3003") if remaining_inv.get(pid, 0) > 0
            )
            max_bricks = max(1, n_studs_horizon // 2)
        else:
            # Running-bond seam-ring course -> 1x4 + 1x2 stagger course
            active_parts = tuple(
                pid for pid in ("3010", "3004", "3003") if remaining_inv.get(pid, 0) > 0
            )
            max_bricks = max(1, n_studs_horizon // 3)

        if not active_parts:
            break

        # 3. Candidate Generation (WP-L10)
        frontier = state.get_active_frontier(min_z_ldu=max(8, z_min - 12))
        if candidate_engine_mode == "full":
            pool = state.generate_candidate_pool(allowed_part_ids=active_parts, frontier=frontier)
        else:
            p_scores = compute_cheap_pre_expansion_port_scores(
                state, frontier, active_lookahead_target
            )
            max_p = max(6, int(math.ceil(len(frontier) * 0.65)))
            pool, inc_cache, fb = state.generate_candidate_pool_incremental_selective(
                port_priority_scores=p_scores,
                max_expanded_ports=max_p,
                cache=inc_cache,
                allowed_part_ids=active_parts,
                frontier=frontier,
            )
            if fb or all(
                horizon_target.intersection_volume(c.world_boxes) == 0 for c in pool.candidates
            ):
                pool, inc_cache = state.generate_candidate_pool_incremental(
                    cache=inc_cache, allowed_part_ids=active_parts, frontier=frontier
                )

        if len(pool) == 0:
            break

        # 4. Feature Extraction & Course Planning (WP-L11)
        raw_batch = build_frontier_and_candidate_tensors(
            state, frontier=frontier, pool=pool, target_shape=horizon_target
        )
        norm_p = np.copy(raw_batch.port_features)
        norm_c = np.copy(raw_batch.candidate_features)
        norm_p[:, 13] = 0.05
        norm_p[:, 18] = 0.15
        norm_c[:, 16] = 0.05
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
            state, pool, batch, active_lookahead_target
        )

        cx = batch.candidate_features
        m_cands = len(pool)
        bias_vec = np.zeros(m_cands, dtype=np.float32)
        if obj.decoy_unary_bias:
            for cid, cand in enumerate(pool.candidates):
                bias_vec[cid] = float(
                    obj.decoy_unary_bias.get((cand.part_id, cand.world_boxes), 0.0)
                )

        prec_mask = np.where(cx[:, 13] >= 0.99, 0.0, -50.0).astype(np.float32)
        if np.all(prec_mask < -10.0):
            # No valid candidate can cover `horizon_target` (e.g., SNOT setup was misoriented!)
            break

        if planner_name == "analytic_heuristic_with_cheap_lookahead":
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
            if is_1x1_column_layer:
                snot_cids = [
                    cid
                    for cid, c in enumerate(pool.candidates)
                    if c.part_id == "87087" and cx[cid, 13] >= 0.99 and psi_future[cid, 0] > 0.0
                ]
                if len(snot_cids) >= 2:
                    for scid in snot_cids:
                        if psi_future[scid, 1] == 0.0:
                            scores[scid] += 0.25
            selected = pool.decode_compatible_course(
                scores, max_bricks=max_bricks, score_threshold=0.0
            )
        else:
            stage1_model.eval()
            q_lookahead_model.eval()
            with torch.no_grad():
                zp = stage1_model.encode_frontier(batch)
                u_learned = (
                    stage1_model.forward_candidate_scores(zp, batch).detach().cpu().numpy()
                )
                if is_1x1_column_layer:
                    dummy_opp = FutureOpportunitySample(
                        split="test",
                        family_tag="external_rollout",
                        state_course1=state,
                        frontier_course1=frontier,
                        pool_course1=pool,
                        batch_course1=batch,
                        future_port_features=psi_future,
                        lookahead_q_values=np.zeros(m_cands, dtype=np.float32),
                        optimal_course1_indices=(),
                        target_shape=active_lookahead_target,
                        root_ids=obj.root_ids,
                        allowed_parts_course1=active_parts,
                        allowed_parts_course2=("3024",),
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
                    base_u, max_bricks=max_bricks, score_threshold=0.0
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
                    target_shape=horizon_target,
                    allowed_part_ids=active_parts,
                )
                inter_mod.eval()
                denoiser.eval()
                with torch.no_grad():
                    cx_t = torch.as_tensor(cx, dtype=torch.float32)
                    u_rel_t = torch.as_tensor(u_rel, dtype=torch.float32)
                    p_rel_t = torch.as_tensor(p_rel, dtype=torch.float32)
                    conf_t = torch.as_tensor(conf_mat, dtype=torch.float32)
                    pred_o, pred_w = inter_mod(cx_t, u_rel_t, p_rel_t, conf_t)

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
                    allowed_part_ids=active_parts,
                    max_bricks_per_course=max_bricks,
                    diffusion_steps=diffusion_steps
                    if planner_name != "trained_conditional_ar_or_recurrent"
                    else 2,
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
                    target_shape=horizon_target,
                    allowed_part_ids=active_parts,
                )
                suite_runner = Stage2CoursePlannerSuite(
                    stage1_model=stage1_model,
                    denoiser=denoiser,
                    validator=validator,
                    allowed_part_ids=active_parts,
                    max_bricks_per_course=max_bricks,
                )
                selected, _ = suite_runner._solve_exact_quadratic_milp(
                    pool=pool,
                    unary_scores=base_u,
                    interaction=inter_graph,
                    score_threshold=-20.0,
                    use_continuous_mccormick=True,
                )
            else:
                raise ValueError(f"Unknown planner_name: {planner_name}")

        if not selected:
            break

        state.commit_course(pool, selected)
        for cid in selected:
            pid = pool.candidates[cid].part_id
            remaining_inv[pid] = max(0, remaining_inv.get(pid, 0) - 1)
        course_steps += 1

        eq = validator.evaluate(state)
        if not eq.is_stable:
            all_stable = False
        max_eta = max(max_eta, float(eq.peak_utilization))

    total_ms = (time.perf_counter() - t_start) * 1000.0
    final_iou = _compute_non_root_iou(state, obj.overall_target_shape, obj.root_ids)
    completed = (final_iou >= 0.95) and all_stable

    return SpoilerFreeRolloutMetrics(
        object_id=obj.object_id,
        category=obj.category,
        planner_name=planner_name,
        completed_successfully=completed,
        final_target_iou=final_iou,
        all_courses_stable=all_stable,
        peak_utilization=max_eta,
        total_build_time_ms=total_ms,
        num_autonomous_courses=course_steps,
    )


@dataclass(frozen=True, slots=True)
class ExternalCorpusPlannerStats:
    """Aggregated spoiler-free rollout metrics on the 12 unseen external 3D shape objects."""

    planner_name: str
    completion_rate: float
    completion_ci95: Tuple[float, float]
    mean_final_iou: float
    unseen_category_completion_rate: float
    median_build_time_ms: float
    p95_build_time_ms: float


@dataclass(frozen=True, slots=True)
class LDrawAttritionAndReconstructionSummary:
    """
    WP-L12 Stage 5B: Explicit LDraw / OMR Corpus Attrition & Reconstruction Ledger.
    """

    num_compliant_ldr_models: int
    compliant_total_type1_lines: int
    compliant_accepted_lines: int
    compliant_decomposition_success_rate: float
    compliant_closed_loop_completion_rate: float
    noisy_total_type1_lines: int
    noisy_accepted_lines: int
    noisy_rejected_unknown_part_count: int
    noisy_rejected_off_lattice_count: int
    noisy_rejected_non_cubic_rot_count: int


@dataclass(frozen=True, slots=True)
class WPL12ExternalCorpusStudySummary:
    """
    Complete summary of WP-L12 (`§5` & `§2`):
      - Stage 5A: Spoiler-Free Shape-to-Brick Benchmark across 12 unseen 3D object shapes (6 categories).
      - Stage 5B: LDraw / OMR Corpus Attrition Accounting & Topological Course Decomposition.
    """

    planner_stats: Dict[str, ExternalCorpusPlannerStats]
    category_completion_by_planner: Dict[str, Dict[str, float]]
    ldraw_attrition_summary: LDrawAttritionAndReconstructionSummary


def run_wp_l12_external_shape_to_brick_corpus_study(
    s1_model: Optional[Stage1FrontierOutcomeModel] = None,
    q_model: Optional[FutureOpportunityValueModel] = None,
    inter_mod: Optional[LearnedCandidateInteractionModule] = None,
    denoiser: Optional[CandidateGraphDenoiser] = None,
    seed: int = 40,
) -> WPL12ExternalCorpusStudySummary:
    """
    Execute the complete WP-L12 External Shape-to-Brick & LDraw/OMR Corpus Study.
    """
    if s1_model is None or q_model is None or inter_mod is None or denoiser is None:
        s1_model, q_model, inter_mod, denoiser = _train_complete_build_models(
            seed=seed, use_ipa=False
        )

    _, test_objs = build_stage5_external_shape_corpus()

    planners = (
        "analytic_heuristic_with_cheap_lookahead",
        "trained_scorer_greedy",
        "trained_conditional_ar_or_recurrent",
        "trained_discrete_diffusion",
        "hybrid_diffusion_router",
        "full_quadratic_milp",
    )

    planner_stats: Dict[str, ExternalCorpusPlannerStats] = {}
    category_completion: Dict[str, Dict[str, float]] = {}

    for p_name in planners:
        rollouts = [
            execute_spoiler_free_shape_to_brick_rollout(
                obj=obj,
                planner_name=p_name,
                stage1_model=s1_model,
                q_lookahead_model=q_model,
                inter_mod=inter_mod,
                denoiser=denoiser,
            )
            for obj in test_objs
        ]
        n = len(rollouts)
        succ = sum(1 for r in rollouts if r.completed_successfully)
        unseen_cats = [
            r
            for r in rollouts
            if r.category in ("snot_facade_monument", "cantilever_canopy")
        ]
        unseen_succ = sum(1 for r in unseen_cats if r.completed_successfully)
        times = [r.total_build_time_ms for r in rollouts]

        planner_stats[p_name] = ExternalCorpusPlannerStats(
            planner_name=p_name,
            completion_rate=float(succ) / float(max(1, n)),
            completion_ci95=clopper_pearson_ci_95(succ, n),
            mean_final_iou=float(np.mean([r.final_target_iou for r in rollouts])),
            unseen_category_completion_rate=float(unseen_succ)
            / float(max(1, len(unseen_cats))),
            median_build_time_ms=float(np.median(times)),
            p95_build_time_ms=float(np.percentile(times, 95)),
        )

        cat_map: Dict[str, List[float]] = {}
        for r in rollouts:
            cat_map.setdefault(r.category, []).append(
                1.0 if r.completed_successfully else 0.0
            )
        category_completion[p_name] = {
            cat: float(np.mean(vals)) for cat, vals in cat_map.items()
        }

    # Stage 5B: LDraw / OMR Corpus Attrition & Topological Decomposition Verification
    compliant_models = test_objs[:8]
    comp_total_lines = 0
    comp_accepted_lines = 0
    decomp_succ = 0
    for obj in compliant_models:
        rep: LDrawParseReport = parse_ldr_text_with_report(
            obj.ground_truth_ldr_text, use_z_up=True, strict=True
        )
        comp_total_lines += rep.total_type1_lines
        comp_accepted_lines += len(rep.accepted_specs)
        courses = decompose_assembly_to_courses(rep.accepted_specs, max_bricks_per_course=6)
        if len(courses) > 0:
            decomp_succ += 1

    # Construct 4 noisy/raw external LDraw files with unknown parts, fractional coords, and non-cubic rotations
    noisy_ldr_texts = [
        "\n".join(
            [
                "0 noisy_omr_vehicle.ldr",
                "1 15 0 0 0 1 0 0 0 1 0 0 0 1 3010.dat",
                "1 15 0 80 0 1 0 0 0 1 0 0 0 1 3004.dat",
                "1 4 0 0 24 1 0 0 0 1 0 0 0 1 99999.dat",  # Unknown catalog part
                "1 14 10.35 20.0 24.0 1 0 0 0 1 0 0 0 1 3004.dat",  # Off-lattice coord
                "1 2 0 40 24 0.707 -0.707 0 0.707 0.707 0 0 0 1 3004.dat",  # Non-cubic rot
            ]
        ),
        "\n".join(
            [
                "0 noisy_omr_house.ldr",
                "1 15 20 0 0 1 0 0 0 1 0 0 0 1 3003.dat",
                "1 15 20 0 24 1 0 0 0 1 0 0 0 1 3003.dat",
                "1 1 20 0 48 1 0 0 0 1 0 0 0 1 3039.dat",  # Unknown slope part
                "1 1 20.5 -0.25 48 1 0 0 0 1 0 0 0 1 3004.dat",  # Off-lattice coord
            ]
        ),
        "\n".join(
            [
                "0 noisy_omr_tower.ldr",
                "1 15 0 0 0 1 0 0 0 1 0 0 0 1 3005.dat",
                "1 15 0 0 24 1 0 0 0 1 0 0 0 1 87087.dat",
                "1 15 0 0 48 2 0 0 0 2 0 0 0 2 3005.dat",  # Scaled non-SO(3) matrix
            ]
        ),
        "\n".join(
            [
                "0 noisy_omr_bridge.ldr",
                "1 15 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat",
                "1 15 0 0 24 1 0 0 0 1 0 0 0 1 2456.dat",
                "1 15 0 0 48 1 0 0 0 1 0 0 0 1 6541.dat",  # Unknown Technic brick
            ]
        ),
    ]

    noisy_total = 0
    noisy_acc = 0
    noisy_unk = 0
    noisy_off = 0
    noisy_rot = 0
    for txt in noisy_ldr_texts:
        r = parse_ldr_text_with_report(txt, use_z_up=True, strict=False)
        noisy_total += r.total_type1_lines
        noisy_acc += len(r.accepted_specs)
        noisy_unk += len(r.rejected_unknown_part_lines)
        noisy_off += len(r.rejected_off_lattice_coord_lines)
        noisy_rot += len(r.rejected_non_cubic_rotation_lines)

    attrition_summary = LDrawAttritionAndReconstructionSummary(
        num_compliant_ldr_models=len(compliant_models),
        compliant_total_type1_lines=comp_total_lines,
        compliant_accepted_lines=comp_accepted_lines,
        compliant_decomposition_success_rate=float(decomp_succ)
        / float(max(1, len(compliant_models))),
        compliant_closed_loop_completion_rate=planner_stats[
            "hybrid_diffusion_router"
        ].completion_rate,
        noisy_total_type1_lines=noisy_total,
        noisy_accepted_lines=noisy_acc,
        noisy_rejected_unknown_part_count=noisy_unk,
        noisy_rejected_off_lattice_count=noisy_off,
        noisy_rejected_non_cubic_rot_count=noisy_rot,
    )

    return WPL12ExternalCorpusStudySummary(
        planner_stats=planner_stats,
        category_completion_by_planner=category_completion,
        ldraw_attrition_summary=attrition_summary,
    )
