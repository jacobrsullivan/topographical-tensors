"""
WP-L4: Candidate Interaction Graph & Four-Planner Controlled Architecture
(including Action-Only Binary Candidate-Pool Discrete Diffusion).

Implements:
1. `CandidateInteractionGraph`: Pairwise conflict edges E_conflict and synergy matrix W
   encoding:
   - Orphan 1x1 dead-end port penalties under restricted inventories,
   - Flush face-packing bonuses,
   - Cross-track seam-alignment penalties & biconnected seam-bridging bonuses,
   - Counter-balancing cantilever torque & SNOT cycle-locking synergies.
2. Four controlled planners operating on the exact same `CandidatePool`:
   - `Planner_1` (`OneShotGreedy`): Unary frontier scores + greedy conflict decoding.
   - `Planner_2` (`OneShotExactMILP`): Unary frontier scores + exact HiGHS MILP solver
     (`scipy.optimize.milp`) over port-capacity and hull-conflict constraints.
   - `Planner_3` (`AutoregressiveBeam`): Sequential candidate rollout with beam search (B=4).
   - `Planner_4` (`BinaryCandidateDiffusion` / `Config_D_Brick`): S=6 iterative denoising
     steps over binary selection probabilities x in {0, 1}^M on the Candidate Interaction Graph
     with a learned `CandidateGraphDenoiser` residual and conflict competition projection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import lil_matrix
import torch
import torch.nn as nn

from topo_brick.evaluation.static_equilibrium import (
    EquilibriumResult,
    SparseStaticEquilibriumValidator,
)
from topo_brick.interface.brick_catalog import get_brick_primitive
from topo_brick.interface.ldu_frames import LDUBox
from topo_brick.interface.port_manifold import (
    ActiveConnectionFrontier,
    AssemblyState,
    CandidatePool,
    GlobalPortId,
)
from topo_brick.models.outcome_head import Stage1FrontierOutcomeModel
from topo_brick.representation.interface_fields import (
    CANDIDATE_FEATURE_CHANNELS,
    FrontierGraphBatch,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)


@dataclass(frozen=True, slots=True)
class CandidateInteractionGraph:
    """
    Pairwise interaction structure over a course CandidatePool {c_1, ..., c_M}.
    """

    unary_orphan_penalty: np.ndarray  # [M] >= 0 penalty for leaving unfillable 1x1 target ports
    synergy_matrix: np.ndarray  # [M, M] pairwise synergy/penalty for compatible pairs (i, j)
    conflict_matrix: np.ndarray  # [M, M] binary indicator for (i, j) in E_conflict or port-sharing


@dataclass(frozen=True, slots=True)
class Stage2CourseMetrics:
    """
    Complete execution & wall-clock latency accounting across all Stage 2 / 2.5 / 3 planners.
    """

    planner_name: str
    num_frontier_ports: int
    num_candidates: int
    proposed_indices: Tuple[int, ...]
    committed_indices: Tuple[int, ...]
    committed_instance_ids: Tuple[int, ...]
    equilibrium_result: EquilibriumResult
    prefix_backoff_steps: int
    course_objective_value: float
    t_candidates_ms: float
    t_features_ms: float
    t_encode_ms: float
    t_plan_ms: float
    t_decode_ms: float
    t_validate_ms: float
    t_total_ms: float
    milp_optimality_gap: float = 0.0
    requires_simultaneous_or_temp_support: bool = False


def _boxes_touch_flush_xy(boxes_a: Sequence[LDUBox], boxes_b: Sequence[LDUBox]) -> bool:
    """
    Return True iff boxes_a and boxes_b lie on the same Z course interval and touch
    flush along an X or Y face without intersecting.
    """
    for ba in boxes_a:
        for bb in boxes_b:
            z_overlap = min(ba.zmax, bb.zmax) - max(ba.zmin, bb.zmin)
            if z_overlap <= 0:
                continue
            x_touch = (ba.xmax == bb.xmin or bb.xmax == ba.xmin) and (
                min(ba.ymax, bb.ymax) - max(ba.ymin, bb.ymin) > 0
            )
            y_touch = (ba.ymax == bb.ymin or bb.ymax == ba.ymin) and (
                min(ba.xmax, bb.xmax) - max(ba.xmin, bb.xmin) > 0
            )
            if x_touch or y_touch:
                return True
    return False


def _boxes_have_aligned_cross_seam(boxes_a: Sequence[LDUBox], boxes_b: Sequence[LDUBox]) -> bool:
    """
    Return True iff two parallel adjacent bricks share an identical span along their
    contact axis (forming a continuous aligned cross-seam rather than a staggered bond).
    """
    ba = boxes_a[0]
    bb = boxes_b[0]
    if min(ba.zmax, bb.zmax) - max(ba.zmin, bb.zmin) <= 0:
        return False
    # Side by side in X, sharing identical Y span
    if ba.xmax == bb.xmin or bb.xmax == ba.xmin:
        if ba.ymin == bb.ymin and ba.ymax == bb.ymax:
            return True
    # Side by side in Y, sharing identical X span
    if ba.ymax == bb.ymin or bb.ymax == ba.ymin:
        if ba.xmin == bb.xmin and ba.xmax == bb.xmax:
            return True
    return False


def build_candidate_interaction_graph(
    state: AssemblyState,
    frontier: ActiveConnectionFrontier,
    pool: CandidatePool,
    batch: FrontierGraphBatch,
    target_shape: Optional[TargetShapeBoxes] = None,
    allowed_part_ids: Optional[Sequence[str]] = None,
) -> CandidateInteractionGraph:
    """
    Construct the Candidate Interaction Graph G_cand = ({1..M}, E_conflict, E_synergy).
    """
    m = len(pool)
    unary_orphan = np.zeros(m, dtype=np.float32)
    synergy = np.zeros((m, m), dtype=np.float32)
    conflict = np.zeros((m, m), dtype=np.float32)

    if m == 0:
        return CandidateInteractionGraph(unary_orphan, synergy, conflict)

    for i in range(m):
        for j in pool.conflict_adjacency[i]:
            conflict[i, j] = 1.0

    cand_feats = batch.candidate_features
    target_precisions = cand_feats[:, 13]

    # Identify target frontier ports that have at least one valid target-covering candidate
    target_cands_by_port: Dict[GlobalPortId, Set[int]] = {}
    if target_shape is not None:
        for gpid, c_tuple in pool.port_to_candidates.items():
            valid_c = {cid for cid in c_tuple if target_precisions[cid] >= 0.99}
            if valid_c:
                target_cands_by_port[gpid] = valid_c

    has_1x1_in_inventory = True
    if allowed_part_ids is not None:
        has_1x1_in_inventory = any(pid in ("3005", "3024") for pid in allowed_part_ids)

    # 1. Compute unary orphan-port penalty (when 1x1 filler bricks are unavailable)
    if target_shape is not None and not has_1x1_in_inventory and target_cands_by_port:
        for i, cand_i in enumerate(pool.candidates):
            if target_precisions[i] < 0.99:
                continue
            blocked_i = pool.conflict_adjacency[i] | {i}
            orphaned_ports = 0
            for gpid, valid_c in target_cands_by_port.items():
                if gpid in cand_i.consumed_ports:
                    continue
                if valid_c.issubset(blocked_i):
                    orphaned_ports += 1
            unary_orphan[i] = 4.5 * float(orphaned_ports)

    # Compute grounded support center for cantilever moment vectors
    gx_list: List[float] = []
    gy_list: List[float] = []
    for placed in state.placed_parts.values():
        if placed.is_grounded:
            for b in placed.world_boxes:
                gx_list.append(0.5 * (b.xmin + b.xmax))
                gy_list.append(0.5 * (b.ymin + b.ymax))
    sup_cx = float(np.mean(gx_list)) if gx_list else 0.0
    sup_cy = float(np.mean(gy_list)) if gy_list else 0.0

    moments: List[Tuple[float, float]] = []
    for cand in pool.candidates:
        b0 = cand.world_boxes[0]
        cx = 0.5 * (b0.xmin + b0.xmax)
        cy = 0.5 * (b0.ymin + b0.ymax)
        prim = get_brick_primitive(cand.part_id)
        moments.append(((cx - sup_cx) * prim.mass_grams, (cy - sup_cy) * prim.mass_grams))

    # 2. Compute pairwise synergy matrix W_{ij} for compatible pairs
    for i in range(m):
        cand_i = pool.candidates[i]
        blocked_i = pool.conflict_adjacency[i] | {i}
        for j in range(i + 1, m):
            if conflict[i, j] > 0.5:
                continue
            cand_j = pool.candidates[j]
            w_ij = 0.0

            # (b) Pairwise orphan-port trap penalty under restricted inventory
            pair_orphans = 0
            if target_shape is not None and not has_1x1_in_inventory and target_cands_by_port:
                if target_precisions[i] >= 0.99 and target_precisions[j] >= 0.99:
                    joint_consumed = cand_i.consumed_ports | cand_j.consumed_ports
                    joint_blocked = blocked_i | pool.conflict_adjacency[j] | {j}
                    for gpid, valid_c in target_cands_by_port.items():
                        if gpid in joint_consumed:
                            continue
                        # Only count newly orphaned ports not already orphaned by i or j alone
                        if valid_c.issubset(joint_blocked) and not (
                            valid_c.issubset(blocked_i)
                            or valid_c.issubset(pool.conflict_adjacency[j] | {j})
                        ):
                            pair_orphans += 1
                    w_ij -= 4.5 * float(pair_orphans)

            # (a) Flush packing bonus inside target shape (only if pair does not trap orphans)
            if pair_orphans == 0 and _boxes_touch_flush_xy(
                cand_i.world_boxes, cand_j.world_boxes
            ):
                if target_shape is None or (
                    target_precisions[i] >= 0.99 and target_precisions[j] >= 0.99
                ):
                    w_ij += 1.5

                # Cross-track aligned seam penalty vs staggered seam bonus
                if _boxes_have_aligned_cross_seam(cand_i.world_boxes, cand_j.world_boxes):
                    w_ij -= 1.2
                else:
                    w_ij += 0.8

            # (c) Seam bridging & orthogonal biconnected ring synergy
            if cand_i.closes_structural_seam and cand_j.closes_structural_seam:
                if cand_i.bridged_instances != cand_j.bridged_instances:
                    w_ij += 1.5
                    bi = cand_i.world_boxes[0]
                    bj = cand_j.world_boxes[0]
                    orient_i = (bi.xmax - bi.xmin) > (bi.ymax - bi.ymin)
                    orient_j = (bj.xmax - bj.xmin) > (bj.ymax - bj.ymin)
                    if orient_i != orient_j:
                        w_ij += 2.0  # Orthogonal seam bridging locks both X and Y axes!

            # (d) Counter-balance cantilever & SNOT arch lock synergy
            # Applies when an overhanging candidate's torque is cancelled by an opposing lever arm:
            # ||m_i + m_j||_2 <= 0.35 * (||m_i||_2 + ||m_j||_2)
            has_overhang = cand_feats[i, 11] > 0.01 or cand_feats[j, 11] > 0.01
            if has_overhang:
                mx_i, my_i = moments[i]
                mx_j, my_j = moments[j]
                norm_i = float(np.hypot(mx_i, my_i))
                norm_j = float(np.hypot(mx_j, my_j))
                norm_sum = float(np.hypot(mx_i + mx_j, my_i + my_j))
                if (norm_i + norm_j) > 10.0 and norm_sum <= 0.35 * (norm_i + norm_j):
                    w_ij += 3.5  # Opposing lever arms cancel net bending torque!
            if (cand_feats[i, 11] > 0.15 and cand_j.creates_biconnected_cycle) or (
                cand_feats[j, 11] > 0.15 and cand_i.creates_biconnected_cycle
            ):
                w_ij += 3.0

            synergy[i, j] = w_ij
            synergy[j, i] = w_ij

    return CandidateInteractionGraph(
        unary_orphan_penalty=unary_orphan,
        synergy_matrix=synergy,
        conflict_matrix=conflict,
    )


def evaluate_course_objective(
    selected_indices: Sequence[int],
    unary_scores: np.ndarray,
    interaction_graph: CandidateInteractionGraph,
) -> float:
    """
    Evaluate the full quadratic course objective:
        J(x) = sum_i (s_i - orphan_i) x_i + 0.5 * sum_{i != j} W_{ij} x_i x_j
    """
    idx = list(selected_indices)
    if not idx:
        return 0.0
    total = float(
        np.sum(unary_scores[idx] - interaction_graph.unary_orphan_penalty[idx])
    )
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            total += float(interaction_graph.synergy_matrix[idx[a], idx[b]])
    return total


PAIRWISE_INTERACTION_CHANNELS: int = 8


def extract_pairwise_interaction_features(
    state: AssemblyState,
    frontier: ActiveConnectionFrontier,
    pool: CandidatePool,
    batch: FrontierGraphBatch,
    target_shape: Optional[TargetShapeBoxes] = None,
    allowed_part_ids: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract raw relational feature tensors `(unary_rel [M, 4], pair_rel [M, M, 8], conflict [M, M])`
    for `LearnedCandidateInteractionModule` without applying hand-coded synergy weights.
    """
    m = len(pool)
    unary_rel = np.zeros((m, 4), dtype=np.float32)
    pair_rel = np.zeros((m, m, 8), dtype=np.float32)
    conflict = np.zeros((m, m), dtype=np.float32)
    if m == 0:
        return unary_rel, pair_rel, conflict

    for i in range(m):
        for j in pool.conflict_adjacency[i]:
            conflict[i, j] = 1.0

    cand_feats = batch.candidate_features
    target_precisions = cand_feats[:, 13]

    target_cands_by_port: Dict[GlobalPortId, Set[int]] = {}
    if target_shape is not None:
        for gpid, c_tuple in pool.port_to_candidates.items():
            valid_c = {cid for cid in c_tuple if target_precisions[cid] >= 0.99}
            if valid_c:
                target_cands_by_port[gpid] = valid_c

    has_1x1_in_inventory = True
    if allowed_part_ids is not None:
        has_1x1_in_inventory = any(pid in ("3005", "3024") for pid in allowed_part_ids)

    for i, cand_i in enumerate(pool.candidates):
        orphaned_ports = 0
        if (
            target_shape is not None
            and not has_1x1_in_inventory
            and target_cands_by_port
            and target_precisions[i] >= 0.99
        ):
            blocked_i = pool.conflict_adjacency[i] | {i}
            for gpid, valid_c in target_cands_by_port.items():
                if gpid in cand_i.consumed_ports:
                    continue
                if valid_c.issubset(blocked_i):
                    orphaned_ports += 1
        unary_rel[i, 0] = float(orphaned_ports)
        unary_rel[i, 1] = float(target_precisions[i])
        unary_rel[i, 2] = float(cand_feats[i, 11])
        unary_rel[i, 3] = 0.0 if has_1x1_in_inventory else 1.0

    gx_list: List[float] = []
    gy_list: List[float] = []
    for placed in state.placed_parts.values():
        if placed.is_grounded:
            for b in placed.world_boxes:
                gx_list.append(0.5 * (b.xmin + b.xmax))
                gy_list.append(0.5 * (b.ymin + b.ymax))
    sup_cx = float(np.mean(gx_list)) if gx_list else 0.0
    sup_cy = float(np.mean(gy_list)) if gy_list else 0.0

    moments: List[Tuple[float, float]] = []
    for cand in pool.candidates:
        b0 = cand.world_boxes[0]
        cx = 0.5 * (b0.xmin + b0.xmax)
        cy = 0.5 * (b0.ymin + b0.ymax)
        prim = get_brick_primitive(cand.part_id)
        moments.append(((cx - sup_cx) * prim.mass_grams, (cy - sup_cy) * prim.mass_grams))

    for i in range(m):
        cand_i = pool.candidates[i]
        blocked_i = pool.conflict_adjacency[i] | {i}
        for j in range(i + 1, m):
            if conflict[i, j] > 0.5:
                continue
            cand_j = pool.candidates[j]
            feats_ij = np.zeros(8, dtype=np.float32)

            pair_orphans = 0
            if target_shape is not None and not has_1x1_in_inventory and target_cands_by_port:
                if target_precisions[i] >= 0.99 and target_precisions[j] >= 0.99:
                    joint_consumed = cand_i.consumed_ports | cand_j.consumed_ports
                    joint_blocked = blocked_i | pool.conflict_adjacency[j] | {j}
                    for gpid, valid_c in target_cands_by_port.items():
                        if gpid in joint_consumed:
                            continue
                        if valid_c.issubset(joint_blocked) and not (
                            valid_c.issubset(blocked_i)
                            or valid_c.issubset(pool.conflict_adjacency[j] | {j})
                        ):
                            pair_orphans += 1
            feats_ij[0] = float(pair_orphans)

            if pair_orphans == 0 and _boxes_touch_flush_xy(cand_i.world_boxes, cand_j.world_boxes):
                if target_shape is None or (
                    target_precisions[i] >= 0.99 and target_precisions[j] >= 0.99
                ):
                    feats_ij[1] = 1.0
                if _boxes_have_aligned_cross_seam(cand_i.world_boxes, cand_j.world_boxes):
                    feats_ij[2] = 1.0
                else:
                    feats_ij[3] = 1.0

            if cand_i.closes_structural_seam and cand_j.closes_structural_seam:
                if cand_i.bridged_instances != cand_j.bridged_instances:
                    feats_ij[4] = 1.0
                    bi = cand_i.world_boxes[0]
                    bj = cand_j.world_boxes[0]
                    orient_i = (bi.xmax - bi.xmin) > (bi.ymax - bi.ymin)
                    orient_j = (bj.xmax - bj.xmin) > (bj.ymax - bj.ymin)
                    if orient_i != orient_j:
                        feats_ij[5] = 1.0

            has_overhang = cand_feats[i, 11] > 0.01 or cand_feats[j, 11] > 0.01
            if has_overhang:
                mx_i, my_i = moments[i]
                mx_j, my_j = moments[j]
                norm_i = float(np.hypot(mx_i, my_i))
                norm_j = float(np.hypot(mx_j, my_j))
                norm_sum = float(np.hypot(mx_i + mx_j, my_i + my_j))
                if (norm_i + norm_j) > 10.0 and norm_sum <= 0.35 * (norm_i + norm_j):
                    feats_ij[6] = 1.0
            if (cand_feats[i, 11] > 0.15 and cand_j.creates_biconnected_cycle) or (
                cand_feats[j, 11] > 0.15 and cand_i.creates_biconnected_cycle
            ):
                feats_ij[7] = 1.0

            pair_rel[i, j] = feats_ij
            pair_rel[j, i] = feats_ij

    return unary_rel, pair_rel, conflict


class LearnedCandidateInteractionModule(nn.Module):
    """
    Stage 3A Learned Candidate Interaction Predictor W_θ(c_i, c_j) and o_θ(c_i).
    Learns unary orphan penalties and pairwise course synergies directly from
    supervision trajectories without hard-coded analytic rules at inference time.
    """

    def __init__(
        self, d_cand: int = CANDIDATE_FEATURE_CHANNELS, d_hidden: int = 64
    ) -> None:
        super().__init__()
        self.orphan_mlp = nn.Sequential(
            nn.Linear(d_cand + 4, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
            nn.Softplus(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_cand + 8, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(
        self,
        cand_features: torch.Tensor,  # [M, 18]
        unary_rel: torch.Tensor,  # [M, 4]
        pair_rel: torch.Tensor,  # [M, M, 8]
        conflict_matrix: torch.Tensor,  # [M, M]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = cand_features.shape[0]
        if m == 0:
            return (
                torch.zeros((0,), device=cand_features.device),
                torch.zeros((0, 0), device=cand_features.device),
            )
        u_in = torch.cat([cand_features, unary_rel], dim=-1)
        pred_orphan = self.orphan_mlp(u_in).squeeze(-1) * unary_rel[:, 3]

        hi = cand_features.unsqueeze(1).expand(m, m, -1)
        hj = cand_features.unsqueeze(0).expand(m, m, -1)
        h_sum = hi + hj
        h_diff = torch.abs(hi - hj)
        e_in = torch.cat([h_sum, h_diff, pair_rel], dim=-1)
        raw_w = self.edge_mlp(e_in).squeeze(-1)
        sym_w = 0.5 * (raw_w + raw_w.transpose(0, 1))
        diag_mask = 1.0 - torch.eye(m, device=cand_features.device)
        compat_mask = (1.0 - conflict_matrix) * diag_mask
        return pred_orphan, sym_w * compat_mask


class CandidateGraphDenoiser(nn.Module):
    """
    Lightweight neural denoising network over the Candidate Interaction Graph G_cand
    for `Planner_4` (`BinaryCandidateDiffusion`).
    """

    def __init__(self, d_cand: int = CANDIDATE_FEATURE_CHANNELS, d_hidden: int = 64) -> None:
        super().__init__()
        self.step_emb = nn.Embedding(16, d_hidden)
        self.node_mlp = nn.Sequential(
            nn.Linear(d_cand + 3, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.out_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )
        # Initialize final layer with small weights so gradient flow is non-zero while
        # preserving physics/synergy dominance prior to training
        nn.init.normal_(self.out_head[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.out_head[-1].bias)

    def forward(
        self,
        cand_features: torch.Tensor,  # [M, 18]
        probs_t: torch.Tensor,  # [M]
        synergy_msg: torch.Tensor,  # [M]
        conflict_msg: torch.Tensor,  # [M]
        step_idx: int,
    ) -> torch.Tensor:
        m = cand_features.shape[0]
        if m == 0:
            return probs_t
        inp = torch.cat(
            [
                cand_features,
                probs_t.unsqueeze(-1),
                synergy_msg.unsqueeze(-1),
                conflict_msg.unsqueeze(-1),
            ],
            dim=-1,
        )
        h = self.node_mlp(inp)
        s_t = torch.full((m,), min(step_idx, 15), dtype=torch.long, device=cand_features.device)
        h = h + self.step_emb(s_t)
        return self.out_head(h).squeeze(-1)


class Stage2CoursePlannerSuite:
    """
    Unified runner implementing all controlled Stage 2 / 2.5 / 3 planners on identical
    CandidatePools:
    - `one_shot_greedy` (`Planner_1`)
    - `one_shot_milp` (`Planner_2a`: Exact Additive Unary MILP)
    - `full_quadratic_milp` (`Planner_2b`: Exact Same-Objective Quadratic MILP via McCormick y_ij)
    - `autoregressive_beam` (`Planner_3`: Sequential Beam Search)
    - `iterative_refinement` / `binary_diffusion` (`Planner_4a`: Iterative Graph Refinement)
    - `trained_discrete_diffusion` (`Planner_4b`: Trained Candidate-Set Discrete Diffusion)
    """

    def __init__(
        self,
        stage1_model: Stage1FrontierOutcomeModel,
        denoiser: Optional[CandidateGraphDenoiser] = None,
        learned_interaction_module: Optional[LearnedCandidateInteractionModule] = None,
        validator: Optional[SparseStaticEquilibriumValidator] = None,
        allowed_part_ids: Optional[Sequence[str]] = None,
        max_bricks_per_course: int = 4,
        diffusion_steps: int = 6,
        beam_width: int = 4,
        use_learned_interaction: bool = False,
    ) -> None:
        self.stage1_model = stage1_model
        self.denoiser = denoiser or CandidateGraphDenoiser()
        self.learned_interaction_module = learned_interaction_module
        self.validator = validator or SparseStaticEquilibriumValidator()
        self.allowed_part_ids = list(allowed_part_ids) if allowed_part_ids is not None else None
        self.max_bricks_per_course = max_bricks_per_course
        self.diffusion_steps = diffusion_steps
        self.beam_width = beam_width
        self.use_learned_interaction = use_learned_interaction

    def _solve_exact_quadratic_milp(
        self,
        pool: CandidatePool,
        unary_scores: np.ndarray,
        interaction: CandidateInteractionGraph,
        score_threshold: float = -20.0,
        use_continuous_mccormick: bool = True,
        time_limit_s: Optional[float] = None,
    ) -> Tuple[List[int], float]:
        """
        Planner_2b (`full_quadratic_milp`, Reviewer Round 2 §3.2 & Literature Agenda §6.1 [R32]):
        Solve the exact same quadratic objective J(x) as the interaction-aware planners:
            max_{x, y} sum_i (s_i - orphan_i) x_i + sum_{i < j} W_{ij} y_{ij}
        subject to:
          - port capacity: sum_{i: u in P_i} x_i <= 1
          - pairwise hull conflicts: x_i + x_j <= 1 for (i, j) in E_conflict
          - course cardinality: sum_i x_i <= max_bricks_per_course
          - McCormick linearization for y_{ij} = x_i x_j:
              y_{ij} - x_i <= 0
              y_{ij} - x_j <= 0
              x_i + x_j - y_{ij} <= 1
              0 <= y_{ij} <= 1
        When `use_continuous_mccormick=True` (default, [R32]), only primary candidate variables
        x_i in {0, 1}^M are constrained to be binary while McCormick auxiliaries y_{ij} in [0, 1]
        are continuous (since binary x_i, x_j automatically force y_{ij} in {0, 1} at every integer
        vertex), eliminating n_pairs branch-and-bound integer variables.
        """
        valid_mask = unary_scores > score_threshold
        valid_indices = [int(i) for i in np.nonzero(valid_mask)[0]]
        if not valid_indices:
            return [], 0.0

        m_sub = len(valid_indices)
        orig_to_sub = {orig: sub for sub, orig in enumerate(valid_indices)}
        base_u = unary_scores - interaction.unary_orphan_penalty
        w_syn = interaction.synergy_matrix
        w_conf = interaction.conflict_matrix

        # Enumerate active non-conflicting pairs (sub_a < sub_b) with |W_{ab}| > 1e-5
        active_pairs: List[Tuple[int, int, float]] = []
        for a in range(m_sub):
            orig_a = valid_indices[a]
            for b in range(a + 1, m_sub):
                orig_b = valid_indices[b]
                if w_conf[orig_a, orig_b] > 0.5:
                    continue
                w_val = float(w_syn[orig_a, orig_b])
                if abs(w_val) > 1e-5:
                    active_pairs.append((a, b, w_val))

        n_pairs = len(active_pairs)
        n_vars = m_sub + n_pairs

        # Objective vector c (minimizing -J(x, y))
        c = np.zeros(n_vars, dtype=np.float64)
        for a, orig_a in enumerate(valid_indices):
            c[a] = -float(base_u[orig_a])
        for k, (_, _, w_val) in enumerate(active_pairs):
            c[m_sub + k] = -w_val

        row_cols: List[List[int]] = []
        row_vals: List[List[float]] = []
        rhs: List[float] = []

        # 1. Port capacity constraints
        for c_tuple in pool.port_to_candidates.values():
            sub_cols = [orig_to_sub[cid] for cid in c_tuple if cid in orig_to_sub]
            if len(sub_cols) > 1:
                row_cols.append(sub_cols)
                row_vals.append([1.0] * len(sub_cols))
                rhs.append(1.0)

        # 2. Disjoint-port hull conflict constraints
        for ca, cb in pool.conflict_edges:
            if ca in orig_to_sub and cb in orig_to_sub:
                row_cols.append([orig_to_sub[ca], orig_to_sub[cb]])
                row_vals.append([1.0, 1.0])
                rhs.append(1.0)

        # 3. Course cardinality constraint
        row_cols.append(list(range(m_sub)))
        row_vals.append([1.0] * m_sub)
        rhs.append(float(self.max_bricks_per_course))

        # 4. McCormick linearization constraints for each y_k = x_a * x_b
        for k, (sub_a, sub_b, _) in enumerate(active_pairs):
            y_col = m_sub + k
            # y_k - x_a <= 0
            row_cols.append([y_col, sub_a])
            row_vals.append([1.0, -1.0])
            rhs.append(0.0)
            # y_k - x_b <= 0
            row_cols.append([y_col, sub_b])
            row_vals.append([1.0, -1.0])
            rhs.append(0.0)
            # x_a + x_b - y_k <= 1
            row_cols.append([sub_a, sub_b, y_col])
            row_vals.append([1.0, 1.0, -1.0])
            rhs.append(1.0)

        mat = lil_matrix((len(row_cols), n_vars), dtype=np.float64)
        for r_idx, (cols, vals) in enumerate(zip(row_cols, row_vals)):
            for col, val in zip(cols, vals):
                mat[r_idx, col] = val

        constraints = LinearConstraint(mat.tocsc(), -np.inf, np.array(rhs, dtype=np.float64))
        if use_continuous_mccormick:
            integrality = np.zeros(n_vars, dtype=np.int64)
            integrality[:m_sub] = 1
        else:
            integrality = np.ones(n_vars, dtype=np.int64)
        options = {"time_limit": float(time_limit_s)} if time_limit_s is not None else None
        res = milp(
            c=c,
            integrality=integrality,
            bounds=(np.zeros(n_vars), np.ones(n_vars)),
            constraints=constraints,
            options=options,
        )
        if not res.success or res.x is None:
            return (
                pool.decode_compatible_course(
                    base_u, max_bricks=self.max_bricks_per_course, score_threshold=score_threshold
                ),
                1.0,
            )

        chosen_sub = [sub for sub in range(m_sub) if res.x[sub] > 0.5]
        chosen_orig = [valid_indices[sub] for sub in chosen_sub]
        chosen_orig.sort(key=lambda idx: float(base_u[idx]), reverse=True)
        gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
        return chosen_orig, gap

    def _solve_exact_unary_milp(
        self,
        pool: CandidatePool,
        unary_scores: np.ndarray,
        score_threshold: float = -1e5,
    ) -> List[int]:
        """
        Planner_2: Solve max_{x in {0,1}^M} sum_i s_i x_i subject to:
          - port capacity: sum_{i: u in P_i} x_i <= 1
          - pairwise hull conflicts: x_i + x_j <= 1 for (i, j) in E_conflict
          - course cardinality: sum_i x_i <= max_bricks_per_course
        using HiGHS (`scipy.optimize.milp`).
        """
        valid_mask = unary_scores > score_threshold
        valid_indices = [int(i) for i in np.nonzero(valid_mask)[0]]
        if not valid_indices:
            return []

        m_sub = len(valid_indices)
        orig_to_sub = {orig: sub for sub, orig in enumerate(valid_indices)}
        # milp minimizes c^T x, so negate positive unary scores
        shifted_scores = np.array([max(1e-3, float(unary_scores[i])) for i in valid_indices])
        c = -shifted_scores

        rows: List[List[int]] = []
        rhs: List[float] = []

        # Port capacity constraints
        for c_tuple in pool.port_to_candidates.values():
            sub_cols = [orig_to_sub[cid] for cid in c_tuple if cid in orig_to_sub]
            if len(sub_cols) > 1:
                rows.append(sub_cols)
                rhs.append(1.0)

        # Disjoint-port hull conflict constraints
        for ca, cb in pool.conflict_edges:
            if ca in orig_to_sub and cb in orig_to_sub:
                rows.append([orig_to_sub[ca], orig_to_sub[cb]])
                rhs.append(1.0)

        # Cardinality constraint
        rows.append(list(range(m_sub)))
        rhs.append(float(self.max_bricks_per_course))

        mat = lil_matrix((len(rows), m_sub), dtype=np.float64)
        for r_idx, cols in enumerate(rows):
            for col in cols:
                mat[r_idx, col] = 1.0

        constraints = LinearConstraint(mat.tocsc(), -np.inf, np.array(rhs, dtype=np.float64))
        integrality = np.ones(m_sub, dtype=np.int64)
        res = milp(
            c=c,
            integrality=integrality,
            bounds=(np.zeros(m_sub), np.ones(m_sub)),
            constraints=constraints,
        )
        if not res.success or res.x is None:
            return pool.decode_compatible_course(
                unary_scores, max_bricks=self.max_bricks_per_course, score_threshold=score_threshold
            )

        chosen_sub = [sub for sub, val in enumerate(res.x) if val > 0.5]
        # Sort chosen candidates by descending unary score
        chosen_orig = [valid_indices[sub] for sub in chosen_sub]
        chosen_orig.sort(key=lambda idx: float(unary_scores[idx]), reverse=True)
        return chosen_orig

    def _run_autoregressive_beam(
        self,
        pool: CandidatePool,
        unary_scores: np.ndarray,
        interaction: CandidateInteractionGraph,
        score_threshold: float = -1e5,
    ) -> List[int]:
        """
        Planner_3: Sequential candidate rollout with beam search (beam_width = B).
        """
        valid_cands = [i for i in range(len(pool)) if float(unary_scores[i]) > score_threshold]
        if not valid_cands:
            return []

        # Beam entries: (cumulative_score, tuple_of_selected_indices, frozenset_of_blocked_indices)
        beam: List[Tuple[float, Tuple[int, ...], Set[int]]] = [(0.0, (), set())]

        for _ in range(self.max_bricks_per_course):
            next_beam: List[Tuple[float, Tuple[int, ...], Set[int]]] = []
            expanded_any = False
            seen_sets: Set[FrozenSet[int]] = set()

            for cum_score, seq, blocked in beam:
                for cid in valid_cands:
                    if cid in blocked:
                        continue
                    cand_set_key = frozenset(seq + (cid,))
                    if cand_set_key in seen_sets:
                        continue
                    seen_sets.add(cand_set_key)

                    step_gain = float(unary_scores[cid] - interaction.unary_orphan_penalty[cid])
                    for prev_id in seq:
                        step_gain += float(interaction.synergy_matrix[prev_id, cid])
                    new_score = cum_score + step_gain
                    new_blocked = set(blocked)
                    new_blocked.add(cid)
                    new_blocked.update(pool.conflict_adjacency[cid])
                    next_beam.append((new_score, seq + (cid,), new_blocked))
                    expanded_any = True

            if not expanded_any:
                break
            next_beam.sort(key=lambda item: item[0], reverse=True)
            beam = next_beam[: self.beam_width]

        best_seq = max(beam, key=lambda item: item[0])[1]
        return list(best_seq)

    @torch.no_grad()
    def _run_binary_diffusion(
        self,
        pool: CandidatePool,
        batch: FrontierGraphBatch,
        unary_scores: np.ndarray,
        interaction: CandidateInteractionGraph,
        score_threshold: float = -1e5,
    ) -> List[int]:
        """
        Planner_4 (`Config_D_Brick`): S-step Binary Candidate-Pool Discrete Diffusion over
        x in {0, 1}^M on the Candidate Interaction Graph G_cand.
        """
        m = len(pool)
        if m == 0:
            return []

        self.denoiser.eval()
        device = next(self.denoiser.parameters()).device
        cand_x = torch.as_tensor(batch.candidate_features, dtype=torch.float32, device=device)

        valid_mask = unary_scores > score_threshold
        base_unary = unary_scores - interaction.unary_orphan_penalty
        w_syn = interaction.synergy_matrix
        w_conf = interaction.conflict_matrix

        # Initialize selection probabilities p^{(S)} in [0, 1]^M
        probs = np.where(valid_mask, 0.5, 0.0).astype(np.float32)
        logits = np.where(valid_mask, base_unary, -1e9).astype(np.float32)

        for step_s in range(self.diffusion_steps, 0, -1):
            # 1. Expected pairwise synergy message from compatible course neighbors
            syn_msg = w_syn @ probs  # [M]

            # 2. Conflict competition pressure from overlapping/shared-port rivals
            # Only rivals with higher or comparable current logits suppress candidate i
            pos_logits = np.maximum(0.0, logits - np.min(logits[valid_mask]) if np.any(valid_mask) else logits)
            conf_msg = (w_conf @ (probs * pos_logits)) / (np.max(pos_logits) + 1e-6)

            # 3. Learned neural denoising residual
            probs_t = torch.as_tensor(probs, dtype=torch.float32, device=device)
            syn_t = torch.as_tensor(syn_msg, dtype=torch.float32, device=device)
            conf_t = torch.as_tensor(conf_msg, dtype=torch.float32, device=device)
            neural_delta = (
                self.denoiser(cand_x, probs_t, syn_t, conf_t, step_s)
                .detach()
                .cpu()
                .numpy()
            )

            # 4. Reverse diffusion logit update with annealed conflict sharpening
            progress = float(self.diffusion_steps - step_s + 1) / float(self.diffusion_steps)
            logits = (
                base_unary
                + (0.6 + 0.6 * progress) * syn_msg
                - (0.4 + 0.8 * progress) * conf_msg
                + 0.1 * neural_delta
            )
            logits = np.where(valid_mask, logits, -1e9)

            # Update soft probabilities via neighborhood competition softmax
            temp = max(0.25, 1.0 - 0.75 * progress)
            scaled = np.clip((logits - np.max(logits[valid_mask] if np.any(valid_mask) else 0.0)) / temp, -30.0, 30.0)
            exp_l = np.where(valid_mask, np.exp(scaled), 0.0)
            denom = exp_l + (w_conf @ exp_l) + 1e-8
            probs = exp_l / denom

        # Final interaction-conditioned projection onto {0, 1}^M (DIFUSCO-style conditional rounding)
        cond_logits = logits.copy()
        selected: List[int] = []
        blocked: Set[int] = set()

        for _ in range(self.max_bricks_per_course):
            best_cid = -1
            best_val = score_threshold
            for cid in range(m):
                if not valid_mask[cid] or cid in blocked:
                    continue
                if float(cond_logits[cid]) > best_val:
                    best_val = float(cond_logits[cid])
                    best_cid = cid
            if best_cid < 0:
                break
            selected.append(best_cid)
            blocked.add(best_cid)
            blocked.update(pool.conflict_adjacency[best_cid])
            # Condition remaining candidate logits on selected candidate via synergy matrix W
            cond_logits = cond_logits + w_syn[:, best_cid]

        # Local 1-brick swap refinement on the interaction graph if an early greedy choice trapped orphans
        best_obj = evaluate_course_objective(selected, unary_scores, interaction)
        for drop_idx in range(len(selected)):
            seed_set = [c for idx, c in enumerate(selected) if idx != drop_idx]
            seed_blocked: Set[int] = set()
            for c in seed_set:
                seed_blocked.add(c)
                seed_blocked.update(pool.conflict_adjacency[c])
            # Greedily complete from seed_set using full quadratic marginal gain
            trial_set = list(seed_set)
            trial_blocked = set(seed_blocked)
            while len(trial_set) < self.max_bricks_per_course:
                best_add = -1
                best_gain = -1e9
                for cid in range(m):
                    if not valid_mask[cid] or cid in trial_blocked:
                        continue
                    gain = float(base_unary[cid]) + sum(
                        float(w_syn[cid, prev]) for prev in trial_set
                    )
                    if gain > best_gain and gain > score_threshold:
                        best_gain = gain
                        best_add = cid
                if best_add < 0:
                    break
                trial_set.append(best_add)
                trial_blocked.add(best_add)
                trial_blocked.update(pool.conflict_adjacency[best_add])
            trial_obj = evaluate_course_objective(trial_set, unary_scores, interaction)
            if trial_obj > best_obj + 1e-4:
                selected = trial_set
                best_obj = trial_obj

        return selected

    @torch.no_grad()
    def _run_hybrid_course_router(
        self,
        pool: CandidatePool,
        batch: FrontierGraphBatch,
        unary_scores: np.ndarray,
        interaction: CandidateInteractionGraph,
        score_threshold: float = -1e5,
        orphan_threshold: float = 0.25,
        synergy_threshold: float = 2.5,
    ) -> Tuple[List[int], bool]:
        """
        Stage 5 (`WP-L11`, Literature Agenda Gap 5): Transparent Hybrid Course Router.
        Inspects the active course candidate subgraph (valid candidates with `unary_scores > score_threshold`):
          - **Fast Regime (`routed_to_diffusion = False`)**: When the valid subproblem has `M_valid <= 4`
            or exhibits neither restricted-inventory orphan penalties (`max o_i <= orphan_threshold`)
            nor strong multi-brick coupling (`max |W_ij| <= synergy_threshold`), decodes directly via
            1-pass interaction-conditioned projection + local 1-swap (`O(M)`), skipping iterative
            D3PM neural denoiser steps.
          - **Coupled Regime (`routed_to_diffusion = True`)**: When orphan-port traps or strong
            pairwise synergy/conflict coupling are present, dispatches to `_run_binary_diffusion`
            (`S = self.diffusion_steps`).
        """
        m = len(pool)
        if m == 0:
            return [], False

        valid_mask = unary_scores > score_threshold
        valid_ids = [int(i) for i in np.nonzero(valid_mask)[0]]
        if not valid_ids:
            return [], False

        if self.max_bricks_per_course <= 1 or len(valid_ids) <= 4:
            base_u = unary_scores - interaction.unary_orphan_penalty
            return (
                pool.decode_compatible_course(
                    base_u, max_bricks=self.max_bricks_per_course, score_threshold=score_threshold
                ),
                False,
            )

        max_orphan = float(np.max(interaction.unary_orphan_penalty[valid_ids]))
        sub_w = interaction.synergy_matrix[np.ix_(valid_ids, valid_ids)]
        max_abs_w = float(np.max(np.abs(sub_w))) if sub_w.size > 0 else 0.0

        # If coupled via orphan traps or strong negative/positive pairwise synergy, route to D3PM
        if max_orphan > orphan_threshold or max_abs_w > synergy_threshold:
            selected = self._run_binary_diffusion(
                pool=pool,
                batch=batch,
                unary_scores=unary_scores,
                interaction=interaction,
                score_threshold=score_threshold,
            )
            return selected, True

        # Fast 1-pass interaction-conditioned projection for weakly coupled courses
        base_unary = unary_scores - interaction.unary_orphan_penalty
        w_syn = interaction.synergy_matrix
        cond_logits = np.where(valid_mask, base_unary, -1e9).astype(np.float32)
        selected: List[int] = []
        blocked: Set[int] = set()

        for _ in range(self.max_bricks_per_course):
            best_cid = -1
            best_val = score_threshold
            for cid in valid_ids:
                if cid in blocked:
                    continue
                if float(cond_logits[cid]) > best_val:
                    best_val = float(cond_logits[cid])
                    best_cid = cid
            if best_cid < 0:
                break
            selected.append(best_cid)
            blocked.add(best_cid)
            blocked.update(pool.conflict_adjacency[best_cid])
            cond_logits = cond_logits + w_syn[:, best_cid]

        return selected, False

    @torch.no_grad()
    def step_course(
        self,
        state: AssemblyState,
        planner_type: str = "binary_diffusion",
        target_shape: Optional[TargetShapeBoxes] = None,
        require_target_precision: float = 0.0,
        custom_unary_bias: Optional[np.ndarray] = None,
    ) -> Stage2CourseMetrics:
        """
        Execute one course-planning step using the specified planner:
        `planner_type` in {"one_shot_greedy", "one_shot_milp", "autoregressive_beam", "binary_diffusion"}.
        """
        self.stage1_model.eval()
        t_start = time.perf_counter()

        # 1. Candidate generation
        t0 = time.perf_counter()
        frontier = state.get_active_frontier()
        pool = state.generate_candidate_pool(
            allowed_part_ids=self.allowed_part_ids, frontier=frontier
        )
        t_candidates_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Feature & Candidate Interaction Graph construction
        t0 = time.perf_counter()
        batch = build_frontier_and_candidate_tensors(
            state=state,
            frontier=frontier,
            pool=pool,
            target_shape=target_shape,
        )
        if self.use_learned_interaction and self.learned_interaction_module is not None:
            self.learned_interaction_module.eval()
            u_rel, p_rel, conf_mat = extract_pairwise_interaction_features(
                state=state,
                frontier=frontier,
                pool=pool,
                batch=batch,
                target_shape=target_shape,
                allowed_part_ids=self.allowed_part_ids,
            )
            dev = next(self.learned_interaction_module.parameters()).device
            pred_o, pred_w = self.learned_interaction_module(
                torch.as_tensor(batch.candidate_features, dtype=torch.float32, device=dev),
                torch.as_tensor(u_rel, dtype=torch.float32, device=dev),
                torch.as_tensor(p_rel, dtype=torch.float32, device=dev),
                torch.as_tensor(conf_mat, dtype=torch.float32, device=dev),
            )
            interaction = CandidateInteractionGraph(
                unary_orphan_penalty=pred_o.detach().cpu().numpy(),
                synergy_matrix=pred_w.detach().cpu().numpy(),
                conflict_matrix=conf_mat,
            )
        else:
            interaction = build_candidate_interaction_graph(
                state=state,
                frontier=frontier,
                pool=pool,
                batch=batch,
                target_shape=target_shape,
                allowed_part_ids=self.allowed_part_ids,
            )
        t_features_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Frontier encoding & unary scoring
        t0 = time.perf_counter()
        z_ports = self.stage1_model.encode_frontier(batch)
        unary_t = self.stage1_model.forward_candidate_scores(z_ports, batch)
        unary_np = unary_t.detach().cpu().numpy().copy()
        if custom_unary_bias is not None and len(custom_unary_bias) == len(unary_np):
            unary_np = unary_np + custom_unary_bias
        if target_shape is not None and require_target_precision > 0.0:
            precisions = batch.candidate_features[:, 13]
            unary_np = np.where(precisions >= require_target_precision, unary_np, -1e9)
        t_encode_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Planner execution & decoding
        t0 = time.perf_counter()
        milp_gap = 0.0
        if planner_type == "one_shot_greedy":
            proposed = pool.decode_compatible_course(
                scores=unary_np,
                max_bricks=self.max_bricks_per_course,
                score_threshold=-1e5,
            )
            t_plan_ms = 0.01
            t_decode_ms = (time.perf_counter() - t0) * 1000.0
        elif planner_type == "one_shot_milp":
            proposed = self._solve_exact_unary_milp(
                pool=pool, unary_scores=unary_np, score_threshold=-1e5
            )
            t_plan_ms = (time.perf_counter() - t0) * 1000.0
            t_decode_ms = 0.01
        elif planner_type == "full_quadratic_milp":
            proposed, milp_gap = self._solve_exact_quadratic_milp(
                pool=pool,
                unary_scores=unary_np,
                interaction=interaction,
                score_threshold=-20.0,
            )
            t_plan_ms = (time.perf_counter() - t0) * 1000.0
            t_decode_ms = 0.01
        elif planner_type == "autoregressive_beam":
            proposed = self._run_autoregressive_beam(
                pool=pool,
                unary_scores=unary_np,
                interaction=interaction,
                score_threshold=-1e5,
            )
            t_plan_ms = (time.perf_counter() - t0) * 1000.0
            t_decode_ms = 0.01
        elif planner_type == "hybrid_diffusion_router":
            proposed, _ = self._run_hybrid_course_router(
                pool=pool,
                batch=batch,
                unary_scores=unary_np,
                interaction=interaction,
                score_threshold=-1e5,
            )
            t_plan_ms = (time.perf_counter() - t0) * 1000.0
            t_decode_ms = 0.01
        elif planner_type in (
            "binary_diffusion",
            "iterative_refinement",
            "trained_discrete_diffusion",
        ):
            proposed = self._run_binary_diffusion(
                pool=pool,
                batch=batch,
                unary_scores=unary_np,
                interaction=interaction,
                score_threshold=-1e5,
            )
            t_plan_ms = (time.perf_counter() - t0) * 1000.0
            t_decode_ms = 0.02
        else:
            raise ValueError(f"Unknown planner_type: {planner_type}")

        # 5. Mechanical validation with prefix backoff & sequential self-support check
        t0 = time.perf_counter()
        pre_commit_state = copy.deepcopy(state)
        committed_indices: List[int] = []
        committed_inst_ids: List[int] = []
        backoff_steps = 0
        eq_res: Optional[EquilibriumResult] = None

        for k in range(len(proposed), 0, -1):
            prefix = proposed[:k]
            trial_state = copy.deepcopy(state)
            trial_state.commit_course(pool, prefix)
            trial_eq = self.validator.evaluate(trial_state)
            if trial_eq.is_stable:
                committed_inst_ids = state.commit_course(pool, prefix)
                committed_indices = prefix
                eq_res = trial_eq
                break
            backoff_steps += 1

        if eq_res is None:
            eq_res = self.validator.evaluate(state)

        # Check if a jointly stable multi-brick course has at least one sequentially self-supporting
        # first brick (Reviewer Round 2 §4.2: batch stability vs. sequential execution)
        requires_temp_support = False
        if len(committed_indices) >= 2 and eq_res.is_stable:
            any_singleton_stable = False
            for cid in committed_indices:
                single_trial = copy.deepcopy(pre_commit_state)
                single_trial.commit_course(pool, [cid])
                if self.validator.evaluate(single_trial).is_stable:
                    any_singleton_stable = True
                    break
            requires_temp_support = not any_singleton_stable

        t_validate_ms = (time.perf_counter() - t0) * 1000.0
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        obj_val = evaluate_course_objective(committed_indices, unary_np, interaction)
        return Stage2CourseMetrics(
            planner_name=planner_type,
            num_frontier_ports=len(frontier),
            num_candidates=len(pool),
            proposed_indices=tuple(proposed),
            committed_indices=tuple(committed_indices),
            committed_instance_ids=tuple(committed_inst_ids),
            equilibrium_result=eq_res,
            prefix_backoff_steps=backoff_steps,
            course_objective_value=obj_val,
            t_candidates_ms=t_candidates_ms,
            t_features_ms=t_features_ms,
            t_encode_ms=t_encode_ms,
            t_plan_ms=t_plan_ms,
            t_decode_ms=t_decode_ms,
            t_validate_ms=t_validate_ms,
            t_total_ms=t_total_ms,
            milp_optimality_gap=milp_gap,
            requires_simultaneous_or_temp_support=requires_temp_support,
        )
