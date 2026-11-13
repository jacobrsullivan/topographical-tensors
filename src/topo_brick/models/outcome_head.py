"""
WP-L3: Stage 1 One-Shot Frontier Scorer, Rapid Stability Predictor,
and End-to-End Course Planner with Sparse Static-Equilibrium Certification.

Implements the complete Stage 1 architecture recommended in peer review (§8 & §9):
1. Encodes the Active Connection Frontier F_t once via `SparsePortIPAEncoder` or
   `RelativeFrameGNNEncoder`.
2. Predicts assembly stability / peak joint utilization in < 2 ms (`predict_stability`).
3. Scores all verified one-course candidates {c_1, ..., c_M} (`score_candidates`).
4. Decodes a mutually compatible course x in {0, 1}^M under exact port-capacity and
   whole-hull conflict constraints (E_conflict).
5. Certifies static mechanical equilibrium via `SparseStaticEquilibriumValidator`
   (automatically backing off to a mechanically stable prefix if a multi-brick course
   overloads a buried joint), and logs the complete wall-clock breakdown T_total.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

from topo_brick.evaluation.static_equilibrium import (
    EquilibriumResult,
    SparseStaticEquilibriumValidator,
)
from topo_brick.interface.port_manifold import (
    AssemblyState,
    CandidatePool,
)
from topo_brick.representation.interface_fields import (
    CANDIDATE_FEATURE_CHANNELS,
    FRONTIER_PORT_CHANNELS,
    FrontierGraphBatch,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)
from topo_brick.models.sparse_port_ipa import (
    RelativeFrameGNNEncoder,
    SparsePortIPAEncoder,
)


@dataclass(frozen=True, slots=True)
class StabilityPrediction:
    """Fast neural readout of assembly stability and per-port load stress."""

    stability_prob: float
    predicted_peak_utilization: float
    port_stress_scores: np.ndarray  # [N_ports]
    inference_time_ms: float


@dataclass(frozen=True, slots=True)
class CourseStepMetrics:
    """
    Complete per-course execution and latency accounting (Reviewer §9):
    T_total = T_candidates + T_features + T_encode + T_score + T_decode + T_validate.
    """

    num_frontier_ports: int
    num_candidates: int
    num_conflict_edges: int
    proposed_candidate_indices: Tuple[int, ...]
    committed_candidate_indices: Tuple[int, ...]
    committed_instance_ids: Tuple[int, ...]
    equilibrium_result: EquilibriumResult
    prefix_backoff_steps: int
    t_candidates_ms: float
    t_features_ms: float
    t_encode_ms: float
    t_score_ms: float
    t_decode_ms: float
    t_validate_ms: float
    t_total_ms: float


class Stage1FrontierOutcomeModel(nn.Module):
    """
    Unified Stage 1 model combining:
    - A relative-frame frontier trunk (`SparsePortIPAEncoder` or `RelativeFrameGNNEncoder`)
    - Rapid stability & peak-utilization readout head (< 2 ms)
    - One-shot candidate course scoring head over {c_1, ..., c_M}
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        use_ipa: bool = True,
        use_analytic_priors: bool = True,
        neural_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_ipa = use_ipa
        self.use_analytic_priors = use_analytic_priors
        self.neural_scale = neural_scale
        if use_ipa:
            self.encoder: nn.Module = SparsePortIPAEncoder(
                in_channels=FRONTIER_PORT_CHANNELS,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
            )
        else:
            self.encoder = RelativeFrameGNNEncoder(
                in_channels=FRONTIER_PORT_CHANNELS,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
            )

        # Global stability & per-port stress head
        self.port_stress_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.global_stability_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 2),  # [stability_logit, log_peak_utilization]
        )

        # Candidate scoring head: pooled port embeddings + 18 candidate features
        self.cand_mlp = nn.Sequential(
            nn.Linear(d_model + CANDIDATE_FEATURE_CHANNELS, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )

    def encode_frontier(
        self, batch: FrontierGraphBatch, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Encode the Active Connection Frontier F_t into node embeddings Z in R^{N x d_model}."""
        if device is None:
            device = next(self.parameters()).device
        port_x = torch.as_tensor(batch.port_features, dtype=torch.float32, device=device)
        edge_idx = torch.as_tensor(batch.edge_index, dtype=torch.long, device=device)
        edge_rot = torch.as_tensor(batch.edge_rel_rot_id, dtype=torch.long, device=device)
        edge_pos = torch.as_tensor(batch.edge_rel_pos_ldu, dtype=torch.float32, device=device)
        edge_attr = torch.as_tensor(batch.edge_attr, dtype=torch.float32, device=device)
        return self.encoder(port_x, edge_idx, edge_rot, edge_pos, edge_attr)

    def forward_stability(
        self, z_ports: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given frontier embeddings `z_ports` [N, d_model], return:
        (stability_logit, predicted_peak_eta, port_stress_scores [N])
        """
        if z_ports.shape[0] == 0:
            zero = torch.zeros((), dtype=z_ports.dtype, device=z_ports.device)
            return zero + 5.0, zero, torch.zeros((0,), dtype=z_ports.dtype, device=z_ports.device)

        port_stress = torch.sigmoid(self.port_stress_head(z_ports).squeeze(-1))
        z_mean = z_ports.mean(dim=0)
        z_max = z_ports.max(dim=0).values
        z_pool = torch.cat([z_mean, z_max], dim=-1)
        out = self.global_stability_head(z_pool)
        stab_logit = out[0]
        pred_eta = nn.functional.softplus(out[1])
        return stab_logit, pred_eta, port_stress

    @torch.no_grad()
    def predict_stability(self, batch: FrontierGraphBatch) -> StabilityPrediction:
        """Single-pass (< 2 ms) neural stability prediction on a FrontierGraphBatch."""
        self.eval()
        t0 = time.perf_counter()
        z_ports = self.encode_frontier(batch)
        stab_logit, pred_eta, port_stress = self.forward_stability(z_ports)
        prob = float(torch.sigmoid(stab_logit).item())
        eta_val = float(pred_eta.item())
        stress_np = port_stress.detach().cpu().numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return StabilityPrediction(
            stability_prob=prob,
            predicted_peak_utilization=eta_val,
            port_stress_scores=stress_np,
            inference_time_ms=elapsed_ms,
        )

    def forward_candidate_scores(
        self, z_ports: torch.Tensor, batch: FrontierGraphBatch
    ) -> torch.Tensor:
        """
        Compute utility scores s in R^M for all candidates in `batch`.
        Combines learned frontier-conditioned MLP logits with the structural/target-shape
        topographical potential channels (target precision, target volume gain, seam closure,
        biconnected cycle creation, and overhang penalty).
        """
        device = z_ports.device
        cand_x = torch.as_tensor(batch.candidate_features, dtype=torch.float32, device=device)
        m_cands = cand_x.shape[0]
        if m_cands == 0:
            return torch.zeros((0,), dtype=torch.float32, device=device)

        pooled_ports = torch.zeros((m_cands, self.d_model), dtype=z_ports.dtype, device=device)
        for cid, p_indices in enumerate(batch.candidate_port_indices):
            if p_indices:
                idx_t = torch.as_tensor(p_indices, dtype=torch.long, device=device)
                pooled_ports[cid] = z_ports[idx_t].mean(dim=0)

        joint = torch.cat([pooled_ports, cand_x], dim=-1)
        raw_neural = self.cand_mlp(joint).squeeze(-1)
        learned_delta = self.neural_scale * raw_neural

        if not self.use_analytic_priors:
            return learned_delta

        # Structural & Target-Shape Topographical Prior:
        # Ch 4: consumed port count, Ch 6: closes_seam, Ch 7: creates_biconnected_cycle,
        # Ch 11: overhang distance, Ch 13: target precision, Ch 14: target volume gain,
        # Ch 15: outside-target volume penalty
        topo_prior = (
            4.0 * cand_x[:, 13]  # target precision
            + 3.0 * cand_x[:, 14]  # target volume gain
            - 6.0 * cand_x[:, 15]  # outside-target protrusion penalty
            + 1.5 * cand_x[:, 6]  # seam-bridging bonus
            + 2.0 * cand_x[:, 7]  # biconnected cycle closure bonus
            + 0.8 * cand_x[:, 4]  # multi-stud lock bonus
            - 1.2 * cand_x[:, 11]  # cantilever overhang penalty
        )
        return topo_prior + learned_delta

    def score_candidates(self, batch: FrontierGraphBatch) -> torch.Tensor:
        """Encode the frontier and compute utility scores s in R^M for all candidates in `batch`."""
        z_ports = self.encode_frontier(batch)
        return self.forward_candidate_scores(z_ports, batch)

    def parameter_inventory(self) -> Dict[str, int]:
        """
        Return exact parameter counts per submodule (Reviewer Round 2 §2).
        """
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        stress_params = sum(p.numel() for p in self.port_stress_head.parameters())
        stab_params = sum(p.numel() for p in self.global_stability_head.parameters())
        cand_params = sum(p.numel() for p in self.cand_mlp.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        return {
            "encoder": enc_params,
            "port_stress_head": stress_params,
            "global_stability_head": stab_params,
            "cand_mlp": cand_params,
            "scoring_and_stability_heads_total": stress_params + stab_params + cand_params,
            "total": total_params,
        }


class Stage1FrontierPlanner:
    """
    End-to-End Stage 1 One-Shot Frontier Planner with exact geometric conflict decoding
    and sparse contact-wrench static equilibrium certification.
    """

    def __init__(
        self,
        model: Stage1FrontierOutcomeModel,
        validator: Optional[SparseStaticEquilibriumValidator] = None,
        allowed_part_ids: Optional[Sequence[str]] = None,
        max_bricks_per_course: int = 4,
    ) -> None:
        self.model = model
        self.validator = validator or SparseStaticEquilibriumValidator()
        self.allowed_part_ids = list(allowed_part_ids) if allowed_part_ids is not None else None
        self.max_bricks_per_course = max_bricks_per_course

    @torch.no_grad()
    def step_course(
        self,
        state: AssemblyState,
        target_shape: Optional[TargetShapeBoxes] = None,
        require_target_precision: float = 0.0,
    ) -> CourseStepMetrics:
        """
        Execute one complete course-planning step on `state`:
        1. Derive verified candidates {c_1, ..., c_M} and conflict graph E_conflict.
        2. Compute 24-ch frontier and 18-ch candidate topographical features.
        3. Encode frontier F_t and score candidates.
        4. Decode compatible course x in {0, 1}^M.
        5. Validate static equilibrium via `SparseStaticEquilibriumValidator` (backing off
           prefix length if shared buried joints are overloaded) and commit to `state`.
        """
        self.model.eval()
        t_start = time.perf_counter()

        # 1. Candidate generation
        t0 = time.perf_counter()
        frontier = state.get_active_frontier()
        pool: CandidatePool = state.generate_candidate_pool(
            allowed_part_ids=self.allowed_part_ids, frontier=frontier
        )
        t_candidates_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Feature computation
        t0 = time.perf_counter()
        batch = build_frontier_and_candidate_tensors(
            state=state,
            frontier=frontier,
            pool=pool,
            target_shape=target_shape,
        )
        t_features_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Frontier encoding
        t0 = time.perf_counter()
        z_ports = self.model.encode_frontier(batch)
        t_encode_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Candidate scoring
        t0 = time.perf_counter()
        scores_t = self.model.forward_candidate_scores(z_ports, batch)
        scores_np = scores_t.detach().cpu().numpy()
        if target_shape is not None and require_target_precision > 0.0:
            precisions = batch.candidate_features[:, 13]
            scores_np = np.where(precisions >= require_target_precision, scores_np, -1e9)
        t_score_ms = (time.perf_counter() - t0) * 1000.0

        # 5. Compatible course decoding (port capacity + E_conflict)
        t0 = time.perf_counter()
        proposed_indices = pool.decode_compatible_course(
            scores=scores_np,
            max_bricks=self.max_bricks_per_course,
            score_threshold=-1e5,
        )
        t_decode_ms = (time.perf_counter() - t0) * 1000.0

        # 6. Mechanical validation with prefix backoff (catches shared buried-joint overload)
        t0 = time.perf_counter()
        committed_indices: List[int] = []
        committed_inst_ids: List[int] = []
        backoff_steps = 0
        eq_res: Optional[EquilibriumResult] = None

        for k in range(len(proposed_indices), 0, -1):
            prefix = proposed_indices[:k]
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
        t_validate_ms = (time.perf_counter() - t0) * 1000.0
        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        return CourseStepMetrics(
            num_frontier_ports=len(frontier),
            num_candidates=len(pool),
            num_conflict_edges=len(pool.conflict_edges),
            proposed_candidate_indices=tuple(proposed_indices),
            committed_candidate_indices=tuple(committed_indices),
            committed_instance_ids=tuple(committed_inst_ids),
            equilibrium_result=eq_res,
            prefix_backoff_steps=backoff_steps,
            t_candidates_ms=t_candidates_ms,
            t_features_ms=t_features_ms,
            t_encode_ms=t_encode_ms,
            t_score_ms=t_score_ms,
            t_decode_ms=t_decode_ms,
            t_validate_ms=t_validate_ms,
            t_total_ms=t_total_ms,
        )
