"""
Opt-in PyTorch learned bottleneck controls (`M1` `DescriptorFrameEncoder`, `M2` `LearnedCatalogAndPortFilter`,
`M3` `SearchDistilledValueModel`) and `BrickActionGrammarAdapter` (`Phase 15`: isolates `torch` from deterministic kernel).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

from topo_brick.interface.brick_catalog import (
    EXTENDED_BRICK_CATALOG,
    STABLETEXT2BRICK_8PART_KIT,
    compute_intrinsic_part_descriptor,
    get_brick_primitive,
)
from topo_brick.interface.port_manifold import (
    ActiveConnectionFrontier,
    AssemblyState,
    CandidatePool,
    IncrementalCandidateCache,
    PortPolarity,
    compute_cheap_pre_expansion_port_scores,
)
from topo_brick.kernel.construction_kernel import TopoBrickKernel
from topo_brick.kernel.course_solvers import _build_blind_grounded_base_pool
from topo_brick.kernel.kernel_contracts import parse_external_record_specs
from topo_brick.representation.interface_fields import (
    SparseTiledTargetField,
    TargetShapeBoxes,
)

# ---------------------------------------------------------------------------
# 6. BrickActionGrammarAdapter (Round 7 Decision 1 & Deliverable 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConstrainedRolloutMetrics:
    """
    Round 7 Decision 1 Constrained Generation Telemetry:
    Separates accepted-action validity (`accepted_action_validity_rate`) from
    100-step rollout completion (`hundred_step_completion_rate`), dead-end rate
    (`dead_end_rate`), early target completion (`early_target_completion_rate`),
    and step latency (`median_step_latency_ms`).
    """

    num_rollouts: int
    accepted_action_validity_rate: float
    hundred_step_completion_rate: float
    early_target_completion_rate: float
    dead_end_rate: float
    mean_steps_completed: float
    median_step_latency_ms: float


class BrickActionGrammarAdapter:
    """
    Stateful grammar constraint mask and action verifier for external autoregressive
    LLM/VLM brick generators (`BrickGPT` / `LegoGPT` / coding agents).
    """

    def __init__(self, kernel: Optional[TopoBrickKernel] = None) -> None:
        self.kernel = kernel or TopoBrickKernel()

    def evaluate_constrained_rollouts(
        self,
        records: Sequence[Dict[str, Any]],
        max_steps: int = 100,
    ) -> ConstrainedRolloutMetrics:
        total_accepted_actions = 0
        valid_accepted_actions = 0
        reached_100_steps = 0
        early_completions = 0
        dead_ends = 0
        steps_list: List[int] = []
        step_latencies: List[float] = []

        for rec in records:
            specs, _ = parse_external_record_specs(
                str(rec["ldr_text"]), allowed_part_ids=set(EXTENDED_BRICK_CATALOG.keys())
            )
            all_boxes = tuple(
                b.transform(sp.pose)
                for sp in specs
                for b in get_brick_primitive(sp.part_id).hull_boxes
            )
            target_field = SparseTiledTargetField(boxes=all_boxes)
            min_z = min(b.zmin for b in all_boxes)

            state = AssemblyState()
            # Seed grounded base blindly from `target_field`
            base_target = target_field.slice_z_band(min_z, min_z + 24)
            base_pool = _build_blind_grounded_base_pool(
                base_target, z0=min_z, dz=24, allowed_part_ids=STABLETEXT2BRICK_8PART_KIT
            )
            sel_b, _ = self.kernel.s1_solver.solve_course(state, base_pool, base_target)
            covered_vol = 0
            for cid_b in sel_b:
                cand_b = base_pool.candidates[cid_b]
                try:
                    state.place_root_part(cand_b.part_id, cand_b.pose, is_grounded=True)
                    covered_vol += int(target_field.intersection_volume(cand_b.world_boxes))
                except ValueError:
                    pass

            target_vol = int(target_field.total_volume_ldu3)
            steps_done = state.num_parts
            hit_dead_end = False
            cand_cache: Optional[IncrementalCandidateCache] = None
            while steps_done < max_steps:
                t_step = time.perf_counter()
                con_res = self.kernel.constrain(
                    state,
                    target_field=target_field,
                    allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
                    max_actions=1,
                    candidate_cache=cand_cache,
                )
                step_latencies.append((time.perf_counter() - t_step) * 1000.0)
                cand_cache = con_res.updated_cache
                if not con_res.valid_actions:
                    certs_now = self.kernel.certify(
                        state, target_field=target_field, verify_prefix=False, verify_insertion=False
                    )
                    if certs_now.geometry.target_iou >= 0.90 or covered_vol >= int(0.95 * target_vol):
                        early_completions += 1
                    else:
                        hit_dead_end = True
                        dead_ends += 1
                    break

                best_act = con_res.valid_actions[0]
                state.place_root_part(best_act.part_id, best_act.pose, is_grounded=False)
                covered_vol += int(best_act.target_intersection_ldu3)
                total_accepted_actions += 1
                if best_act.post_step_utilization <= 1.0:
                    valid_accepted_actions += 1
                steps_done += 1

                # Stop early ONLY when target volume is genuinely covered (>= 98% recall),
                # never via reference brick-count `len(specs)`!
                if covered_vol >= int(0.98 * target_vol):
                    if steps_done < max_steps:
                        early_completions += 1
                    break

            if steps_done >= max_steps and not hit_dead_end:
                reached_100_steps += 1
            steps_list.append(steps_done)

        n_roll = max(len(records), 1)
        return ConstrainedRolloutMetrics(
            num_rollouts=len(records),
            accepted_action_validity_rate=float(valid_accepted_actions / max(total_accepted_actions, 1)),
            hundred_step_completion_rate=float(reached_100_steps / n_roll),
            early_target_completion_rate=float(early_completions / n_roll),
            dead_end_rate=float(dead_ends / n_roll),
            mean_steps_completed=float(np.mean(steps_list)),
            median_step_latency_ms=float(np.median(step_latencies)) if step_latencies else 0.0,
        )


# ---------------------------------------------------------------------------
# 7. Learned Bottleneck Modules (M1, M2, M3) & 3-Part Promotion Gates
# ---------------------------------------------------------------------------


class DescriptorFrameEncoder(nn.Module):
    """
    `M1` Transferable Intrinsic Part Descriptor & Frontier Frame Encoder.
    Uses `compute_intrinsic_part_descriptor(part_id)` (`16` geometry/port-derived features)
    with ZERO per-part-ID lookup table so newly added catalog parts work zero-shot.
    """

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.part_proj = nn.Linear(16, d_model)
        self.port_proj = nn.Linear(24, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, port_features: torch.Tensor, part_descriptors: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.port_proj(port_features) + self.part_proj(part_descriptors))
        return self.out_norm(h)


class LearnedCatalogAndPortFilter(nn.Module):
    """
    `M2` Learned Frontier Port & Catalog Priority Filter.
    Trained on genuine `S1` solver port-utilization labels (`0` formula copying) to rank
    frontier ports prior to `generate_candidate_pool_selective`.
    """

    def __init__(self, d_model: int = 32) -> None:
        super().__init__()
        self.encoder = DescriptorFrameEncoder(d_model=d_model)
        self.head = nn.Linear(d_model, 1)

    def score_ports(
        self,
        state: AssemblyState,
        frontier: ActiveConnectionFrontier,
        target_shape: Optional[object] = None,
    ) -> np.ndarray:
        n_ports = len(frontier)
        if n_ports == 0:
            return np.zeros(0, dtype=np.float32)
        base_prior = compute_cheap_pre_expansion_port_scores(state, frontier, target_shape=target_shape)
        port_feats = np.zeros((n_ports, 24), dtype=np.float32)
        part_descs = np.zeros((n_ports, 16), dtype=np.float32)
        for idx, ep in enumerate(frontier.ports):
            port_feats[idx, 0] = float(ep.polarity)
            port_feats[idx, 1:4] = np.asarray(ep.world_normal, dtype=np.float32)
            port_feats[idx, 4:7] = np.asarray(ep.gravity_local, dtype=np.float32)
            port_feats[idx, 7] = float(min(ep.support_depth, 20)) / 20.0
            port_feats[idx, 8] = float(base_prior[idx]) / 20.0
            part_descs[idx] = compute_intrinsic_part_descriptor(ep.part_id)

        with torch.no_grad():
            h = self.encoder(torch.from_numpy(port_feats), torch.from_numpy(part_descs))
            delta = self.head(h).squeeze(-1).numpy()
        return (base_prior + 0.25 * delta).astype(np.float32)


class SearchDistilledValueModel(nn.Module):
    """
    `M3` Search-Distilled 2-Course Setup Value Model.
    Trained exclusively on genuine 2-course search rollout outcomes (`solve_exact_two_course_teacher`
    Lookahead IoU labels, Round 6 `F4` fix: zero analytical formula copying!).
    Combines `candidate_features` (`18`-D), 4 geometric port-continuation & inventory-scarcity probes,
    and `M1` part-ID-free intrinsic geometric descriptors (`16`-D).
    """

    def __init__(self, in_channels: int = 18, d_hidden: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.probe_channels = 4
        self.net = nn.Sequential(
            nn.Linear(in_channels + self.probe_channels + 16, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def build_input_matrix(
        self,
        pool: CandidatePool,
        candidate_features: np.ndarray,
        target_shape: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
        inventory_counts: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        m = len(pool)
        if m == 0:
            return np.zeros((0, self.in_channels + self.probe_channels + 16), dtype=np.float32)
        part_descs = np.vstack(
            [compute_intrinsic_part_descriptor(c.part_id) for c in pool.candidates]
        ).astype(np.float32)
        probes = np.zeros((m, self.probe_channels), dtype=np.float32)
        tot_t_vol = float(target_shape.total_volume_ldu3) if target_shape is not None else 0.0
        for cid, cand in enumerate(pool.candidates):
            prim = get_brick_primitive(cand.part_id)
            matched_local = {lp for lp, _ in cand.matched_pairs}
            if target_shape is not None:
                vert_hits = 0
                vert_tot = 0
                lat_hits = 0
                for p in prim.ports:
                    if p.port_id in matched_local:
                        continue
                    wf = cand.pose.compose(p.local_frame)
                    pt_out = wf.transform_point((0, 0, 4 if p.polarity == PortPolarity.STUD else -4))
                    hit = target_shape.contains_point(pt_out)
                    if abs(int(wf.z_axis[2])) == 1:
                        vert_tot += 1
                        if hit:
                            vert_hits += 1
                    elif hit:
                        lat_hits += 1
                probes[cid, 0] = float(vert_hits) / float(max(vert_tot, 1))
                probes[cid, 1] = float(lat_hits)
                z_top = max(b.zmax for b in cand.world_boxes)
                upper_vol = float(target_shape.band_volume_ldu3(z_top, z_top + 96))
                probes[cid, 2] = upper_vol / max(tot_t_vol, 1.0)
            if inventory_counts is not None:
                probes[cid, 3] = 1.0 / float(max(inventory_counts.get(cand.part_id, 10), 1))
        return np.hstack(
            [candidate_features[:, : self.in_channels].astype(np.float32), probes, part_descs]
        )

    def predict_candidate_values(
        self,
        pool: CandidatePool,
        candidate_features: np.ndarray,
        target_shape: Optional[SparseTiledTargetField | TargetShapeBoxes] = None,
        inventory_counts: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        m = len(pool)
        if m == 0:
            return np.zeros(0, dtype=np.float32)
        x = self.build_input_matrix(
            pool, candidate_features, target_shape=target_shape, inventory_counts=inventory_counts
        )
        with torch.no_grad():
            out = self.net(torch.from_numpy(x)).squeeze(-1).numpy()
        return out.astype(np.float32)

