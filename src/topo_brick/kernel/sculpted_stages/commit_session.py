"""
Stage 3b: Mutable `SculptedLiveSession` & incremental 6-DoF commit engine (`Phase 15`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_brick.evaluation.static_equilibrium import FoundationPolicy
from topo_brick.interface.brick_catalog import get_brick_primitive
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame
from topo_brick.interface.port_manifold import AssemblyState, SpatialBoxHashIndex
from topo_brick.kernel.insertion_corridors import check_monotone_insertion_corridor
from topo_brick.kernel.sculpted_primitives import _box_hits_forbidden, _fast_rel_mask6


@dataclass(frozen=True, slots=True)
class SessionCommitSnapshot:
    """Measurable pre-stage/pre-commit invariant snapshot of `SculptedLiveSession` (`Phase 16`)."""

    num_parts: int
    occ6_live_total: int
    inter_live: int
    over_live: int
    remaining_stock_total: Optional[int]
    build_mold_counts_total: int
    recorded_steps_count: int
    candidate_poses_scored: int


@dataclass
class SculptedLiveSession:
    """Shared mutable assembly, `occ6_live`, inventory, and LP witness state across sculpted stages."""

    kernel: Any
    prep: Any
    core_plan: Any
    coarse_target_field: Any
    adaptive_target: Any
    allowed_part_ids: Optional[Sequence[str]]
    part_retrieval: str
    retrieval_top_k: int
    license_filter: str
    semantic_style_hint: Optional[str]
    inventory_counts: Optional[Dict[str, int]]
    support_policy: str
    foundation_policy: FoundationPolicy
    forbidden_support_boxes: Sequence[LDUBox]
    colored_inventory: Optional[Any]
    hollow_shell_policy: Optional[Any]
    view_priority: Optional[Any]
    suppress_studless_capping_boxes: Sequence[LDUBox]
    deadline_monotonic_s: Optional[float]
    ground_plane_z_ldu: float
    forced_plan: Optional[Any]
    eval_only: bool
    max_waves: int
    verts_ldu: Optional[np.ndarray] = None
    state: AssemblyState = field(default_factory=AssemblyState)
    rem_inv: Optional[Dict[str, int]] = None
    occ6_live: Any = None
    inter_live: int = 0
    mesh_sub_total: int = 0
    over_live: int = 0
    base_witness: Optional[Any] = None
    base_lp_eta: float = 0.0
    plumb_deltas: Dict[Tuple[int, int], Tuple[float, float, float]] = field(default_factory=dict)
    plumb_hop_cache: Dict[int, Any] = field(default_factory=dict)
    skin_lp_calls: int = 0
    max_skin_lp_calls: int = 22
    consec_skin_lp_fails: int = 0
    deferred_lp_in_batch: bool = False
    batch_new_commits: List[Tuple[str, LDUFrame, bool]] = field(default_factory=list)
    build_mold_counts: Dict[str, int] = field(default_factory=dict)
    is_megastructure_footprint: bool = False
    eval_fast_pose_delta_iou: Optional[Callable[..., float]] = None
    eval_boxes_delta_iou: Optional[Callable[..., Tuple[float, int, int]]] = None
    protrusion_clear_against_existing: Optional[Callable[[str, LDUFrame], bool]] = None
    deadline_expired: Optional[Callable[[], bool]] = None
    try_commit_part: Optional[Callable[..., bool]] = None
    candidate_poses_scored: int = 0

    def capture_invariant_snapshot(self) -> SessionCommitSnapshot:
        """Capture exact volume, stock, trace, and scorer counters before a placement stage."""
        occ_tot = int(np.count_nonzero(self.occ6_live)) if self.occ6_live is not None else 0
        rem_tot = (
            int(sum(int(v) for v in self.rem_inv.values()))
            if self.rem_inv is not None
            else None
        )
        molds_tot = int(sum(int(v) for v in self.build_mold_counts.values()))
        rec_cnt = (
            len(self.state.recorded_step_ids) if self.state.has_execution_trace else 0
        )
        return SessionCommitSnapshot(
            num_parts=int(self.state.num_parts),
            occ6_live_total=occ_tot,
            inter_live=int(self.inter_live),
            over_live=int(self.over_live),
            remaining_stock_total=rem_tot,
            build_mold_counts_total=molds_tot,
            recorded_steps_count=rec_cnt,
            candidate_poses_scored=int(self.candidate_poses_scored),
        )

    def resync_live_metrics(self) -> None:
        """Recompute exact `occ6_live`, `inter_live`, `over_live`, and `rem_inv` from `self.state`."""
        tgt = self.prep.occ6
        gx6, gy6, gz6 = tgt.shape
        occ_new = np.zeros_like(tgt)
        for pl in self.state.placed_parts.values():
            for b in pl.world_boxes:
                ux0 = max(0, int(math.floor(b.xmin / 5.0)))
                ux1 = min(gx6, int(math.ceil(b.xmax / 5.0)))
                uy0 = max(0, int(math.floor(b.ymin / 5.0)))
                uy1 = min(gy6, int(math.ceil(b.ymax / 5.0)))
                uz0 = max(0, int(math.floor(b.zmin / 6.0)))
                uz1 = min(gz6, int(math.ceil(b.zmax / 6.0)))
                if ux1 > ux0 and uy1 > uy0 and uz1 > uz0:
                    occ_new[ux0:ux1, uy0:uy1, uz0:uz1] = True
        self.occ6_live = occ_new
        self.inter_live = int(np.count_nonzero(occ_new & tgt))
        self.over_live = int(np.count_nonzero(occ_new & (~tgt)))
        if self.prep.max_inv_snapshot is not None:
            rem_new = {str(k): int(v) for k, v in self.prep.max_inv_snapshot.items()}
            for pl in self.state.placed_parts.values():
                can_m = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pl.part_id) or pl.part_id
                rem_new[can_m] = rem_new.get(can_m, 0) - 1
            self.rem_inv = rem_new

    def verify_commit_transition(
        self,
        before: SessionCommitSnapshot,
        *,
        allow_pruning: bool = False,
    ) -> None:
        """
        Verify `occ6_live == inter_live + over_live`, non-negative `rem_inv`, exact stock
        conservation against `max_inv_snapshot`, and `recorded_step_ids` sync (`Phase 16`).
        """
        if self.occ6_live is not None:
            tgt = self.prep.occ6
            exp_inter = int(np.count_nonzero(self.occ6_live & tgt))
            exp_over = int(np.count_nonzero(self.occ6_live & (~tgt)))
            if int(self.inter_live) != exp_inter or int(self.over_live) != exp_over:
                raise RuntimeError(
                    f"SculptedLiveSession occ6 invariant drift: "
                    f"tracked=({self.inter_live},{self.over_live}) vs actual=({exp_inter},{exp_over})"
                )
            if not allow_pruning and (exp_inter + exp_over) < before.occ6_live_total:
                raise RuntimeError(
                    f"SculptedLiveSession non-pruning stage shrank occ6_live: "
                    f"{before.occ6_live_total} -> {exp_inter + exp_over}"
                )
        if self.rem_inv is not None:
            if any(int(v) < 0 for v in self.rem_inv.values()):
                raise RuntimeError("SculptedLiveSession rem_inv overrun (< 0)")
            if self.prep.max_inv_snapshot is not None:
                placed_cnt: Dict[str, int] = {}
                for pl in self.state.placed_parts.values():
                    can_m = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pl.part_id) or pl.part_id
                    placed_cnt[can_m] = placed_cnt.get(can_m, 0) + 1
                for m_id, init_qty in self.prep.max_inv_snapshot.items():
                    if int(self.rem_inv.get(m_id, 0)) + placed_cnt.get(m_id, 0) != int(init_qty):
                        raise RuntimeError(
                            f"SculptedLiveSession stock conservation drift for {m_id}: "
                            f"rem={self.rem_inv.get(m_id, 0)} + placed={placed_cnt.get(m_id, 0)} != {init_qty}"
                        )
                for m_id, p_qty in placed_cnt.items():
                    if p_qty > int(self.prep.max_inv_snapshot.get(m_id, 0)):
                        raise RuntimeError(
                            f"SculptedLiveSession placed unbudgeted mold {m_id}: {p_qty} > {self.prep.max_inv_snapshot.get(m_id, 0)}"
                        )
        if self.state.has_execution_trace:
            if len(self.state.recorded_step_ids) != self.state.num_parts:
                raise RuntimeError(
                    f"SculptedLiveSession recorded_step_ids desync: "
                    f"{len(self.state.recorded_step_ids)} != {self.state.num_parts}"
                )
        if int(self.candidate_poses_scored) < before.candidate_poses_scored:
            raise RuntimeError("SculptedLiveSession candidate_poses_scored regressed")


def verify_standalone_sculpted_state_invariants(
    state: AssemblyState,
    prep: Any,
    *,
    forbidden_support_boxes: Sequence[LDUBox] = (),
    ground_plane_z_ldu: float = 0.0,
) -> None:
    """
    Enforce stock, world-datum, forbidden-volume, and execution-trace invariants on standalone
    fast-path `AssemblyState` builds that bypass `SculptedLiveSession` (`Phase 16.1`).
    """
    max_inv = getattr(prep, "max_inv_snapshot", None)
    if max_inv is not None:
        placed_cnt: Dict[str, int] = {}
        for pl in state.placed_parts.values():
            can_m = CATALOG_RETRIEVAL_INDEX.resolve_canonical_mold_id(pl.part_id) or pl.part_id
            placed_cnt[can_m] = placed_cnt.get(can_m, 0) + 1
        for m_id, p_qty in placed_cnt.items():
            lim = int(max_inv.get(m_id, max_inv.get(str(m_id), 0)))
            if p_qty > lim:
                raise RuntimeError(
                    f"Standalone sculpted fast-path exceeded finite stock for {m_id}: {p_qty} > {lim}"
                )
    gz = float(ground_plane_z_ldu)
    occ6_ref = getattr(prep, "occ6", None)
    wb_xmax = float(occ6_ref.shape[0] * 5) if occ6_ref is not None else None
    wb_ymax = float(occ6_ref.shape[1] * 5) if occ6_ref is not None else None
    wb_zmax = gz + float(occ6_ref.shape[2] * 6) if occ6_ref is not None else None
    for pl in state.placed_parts.values():
        for wb in pl.world_boxes:
            if float(wb.zmin) < gz - 0.25:
                raise RuntimeError(
                    f"Standalone sculpted fast-path placed hull below ground datum ({wb.zmin} < {gz})"
                )
            if wb_xmax is not None and (
                float(wb.xmin) < -0.25
                or float(wb.ymin) < -0.25
                or float(wb.xmax) > wb_xmax + 0.25
                or float(wb.ymax) > wb_ymax + 0.25  # type: ignore[operator]
                or float(wb.zmax) > wb_zmax + 0.25  # type: ignore[operator]
            ):
                raise RuntimeError(
                    f"Standalone sculpted fast-path placed hull outside workbench bounds: {wb}"
                )
            for fb in forbidden_support_boxes:
                if wb.intersects(fb):
                    raise RuntimeError(
                        "Standalone sculpted fast-path intersected forbidden_support_boxes"
                    )
    if state.has_execution_trace and len(state.recorded_step_ids) != state.num_parts:
        raise RuntimeError(
            f"Standalone sculpted fast-path recorded_step_ids desync: "
            f"{len(state.recorded_step_ids)} != {state.num_parts}"
        )


def build_and_commit_core_session(sess: SculptedLiveSession) -> None:
    """Initialize `sess` closures and commit planned eroded-core & SNOT courses bottom-up."""
    kernel = sess.kernel
    occ6 = sess.prep.occ6
    occ6_csum = sess.prep.occ6_csum
    gz6_nonempty_limit = sess.prep.gz6_nonempty_limit
    eff_inv_map = sess.prep.eff_inv_map
    kit_ids = sess.prep.kit_ids
    kit_set = sess.prep.kit_set
    n_plates_z = sess.prep.n_plates_z
    view_weight_grid_6 = sess.prep.view_weight_grid_6
    is_hollow_shell_active = sess.prep.is_hollow_shell_active
    foundation_policy = sess.foundation_policy
    forbidden_support_boxes = sess.forbidden_support_boxes
    deadline_monotonic_s = sess.deadline_monotonic_s
    ground_plane_z_ldu = sess.ground_plane_z_ldu

    planned_by_z = sess.core_plan.planned_by_z
    layer_phase = sess.core_plan.layer_phase
    core_occ_pre_sculpt = sess.core_plan.core_occ_pre_sculpt
    is_thin_regime = sess.core_plan.is_thin_regime
    eff_gz6_sky_limit = sess.core_plan.eff_gz6_sky_limit
    w_x = sess.core_plan.w_x
    w_y = sess.core_plan.w_y

    sess.state = AssemblyState()
    if eff_inv_map is not None:
        sess.rem_inv = dict(eff_inv_map)

    sess.occ6_live = np.zeros_like(occ6)
    sess.inter_live = 0
    mesh_sub_total = int(np.count_nonzero(occ6))
    sess.over_live = 0  # `union_live = mesh_sub_total + over_live`!

    sess.base_witness = None
    sess.base_lp_eta = 0.0
    sess.plumb_deltas: Dict[Tuple[int, int], Tuple[float, float, float]] = {}
    sess.plumb_hop_cache: Dict[int, Any] = {}
    sess.candidate_poses_scored = int(getattr(sess.core_plan, "s3_poses_scored", 0))

    def _add_boxes_to_live_occ6(wbs: Sequence[LDUBox]) -> None:
        gx, gy, gz = occ6.shape
        for b in wbs:
            x0 = max(0, int(math.floor(b.xmin / 5.0)))
            x1 = min(gx, int(math.ceil(b.xmax / 5.0)))
            y0 = max(0, int(math.floor(b.ymin / 5.0)))
            y1 = min(gy, int(math.ceil(b.ymax / 5.0)))
            z0 = max(0, int(math.floor(b.zmin / 6.0)))
            z1 = min(gz, int(math.ceil(b.zmax / 6.0)))
            if x1 > x0 and y1 > y0 and z1 > z0:
                cur_sl = sess.occ6_live[x0:x1, y0:y1, z0:z1]
                new_m = ~cur_sl
                if np.any(new_m):
                    tgt_sl = occ6[x0:x1, y0:y1, z0:z1]
                    sess.inter_live += int(np.count_nonzero(new_m & tgt_sl))
                    sess.over_live += int(np.count_nonzero(new_m & (~tgt_sl)))
                    cur_sl[:] = True

    thin_panel_pids: Set[str] = {
        pid_tp
        for pid_tp in kit_ids
        if (ent_tp := CATALOG_RETRIEVAL_INDEX.get_entry(pid_tp)) is not None
        and ent_tp.taxonomy_class == "THIN_PANEL"
    }

    def _eval_fast_pose_delta_iou(
        pid_f: str,
        rid_f: int,
        wx_f: int,
        wy_f: int,
        wz_f: int,
        require_ground_z0: bool = False,
        enforce_min_precision: bool = True,
    ) -> float:
        """
        Ultra-fast (`< 0.5 us` rejected / `~1 us` scored) zero-allocation exact `ΔIoU` pre-filter using
        `@lru_cache` relative `5x5x6 LDU` masks BEFORE constructing `LDUBox` tuples or running collision queries.
        """
        if mesh_sub_total == 0:
            return -1.0
        sess.candidate_poses_scored += 1
        qx, rx_mod = divmod(wx_f, 5)
        qy, ry_mod = divmod(wy_f, 5)
        qz, rz_mod = divmod(wz_f, 6)
        rx0, rx1, ry0, ry1, rz0, rz1, min_z_loc, is_full, rmask = _fast_rel_mask6(
            pid_f, rid_f, rx_mod, ry_mod, rz_mod
        )
        w_zmin = min_z_loc + wz_f
        if require_ground_z0:
            if w_zmin != 0:
                return -1.0
        elif w_zmin < 0:
            return -1.0

        gx, gy, gz = occ6.shape
        x0, x1 = qx + rx0, qx + rx1
        y0, y1 = qy + ry0, qy + ry1
        z0, z1 = qz + rz0, qz + rz1
        # Never clip out-of-bounds overhangs: `occ6` already includes `+20 LDU` (`+4` sub-voxel) upper margin
        if x0 < 0 or y0 < 0 or z0 < 0 or x1 > gx or y1 > gy or z1 > gz:
            return -1.0
        if x1 <= x0 or y1 <= y0 or z1 <= z0 or z0 >= gz6_nonempty_limit:
            return -1.0
        box_tgt_cnt = int(
            occ6_csum[x1, y1, z1]
            - occ6_csum[x0, y1, z1]
            - occ6_csum[x1, y0, z1]
            - occ6_csum[x1, y1, z0]
            + occ6_csum[x0, y0, z1]
            + occ6_csum[x0, y1, z0]
            + occ6_csum[x1, y0, z0]
            - occ6_csum[x0, y0, z0]
        )
        if box_tgt_cnt == 0:
            return -1.0
        if require_ground_z0 and int(
            occ6_csum[x1, y1, 1]
            - occ6_csum[x0, y1, 1]
            - occ6_csum[x1, y0, 1]
            + occ6_csum[x0, y0, 1]
        ) < max(2, ((x1 - x0) * (y1 - y0)) // 4):
            return -1.0

        live_sl = sess.occ6_live[x0:x1, y0:y1, z0:z1]
        if live_sl.all():
            return 0.0
        if is_full:
            live_cnt = int(np.count_nonzero(live_sl))
            tot_new = live_sl.size - live_cnt
            tgt_sl_f = occ6[x0:x1, y0:y1, z0:z1]
            if live_cnt == 0:
                d_in = box_tgt_cnt
                sky_ov_f = (
                    (x1 - x0) * (y1 - y0) * max(0, z1 - max(z0, gz6_nonempty_limit))
                    if z1 > gz6_nonempty_limit
                    else 0
                )
                if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
                    z_sk0 = max(0, eff_gz6_sky_limit - z0)
                    sk_tot = (x1 - x0) * (y1 - y0) * (z1 - max(z0, eff_gz6_sky_limit))
                    sk_in = int(np.count_nonzero(tgt_sl_f[:, :, z_sk0:]))
                    if sk_tot >= 8 and sk_in * 3 < sk_tot:
                        return -1.0
            else:
                new_m = ~live_sl
                d_in = int(np.count_nonzero(new_m & tgt_sl_f))
                if d_in == 0:
                    return -1.0
                sky_ov_f = (
                    int(np.count_nonzero(new_m[:, :, max(0, gz6_nonempty_limit - z0) :]))
                    if z1 > gz6_nonempty_limit
                    else 0
                )
                if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
                    z_sk0 = max(0, eff_gz6_sky_limit - z0)
                    sk_sl = new_m[:, :, z_sk0:]
                    sk_tot = int(np.count_nonzero(sk_sl))
                    if sk_tot >= 8 and int(np.count_nonzero(sk_sl & tgt_sl_f[:, :, z_sk0:])) * 3 < sk_tot:
                        return -1.0
            d_ov = tot_new - d_in
        else:
            new_m = rmask & (~live_sl)
            tot_new = int(np.count_nonzero(new_m))
            if tot_new == 0:
                return 0.0
            tgt_sl_nf = occ6[x0:x1, y0:y1, z0:z1]
            d_in = int(np.count_nonzero(new_m & tgt_sl_nf))
            if d_in == 0:
                return -1.0
            d_ov = tot_new - d_in
            sky_ov_f = (
                int(np.count_nonzero(new_m[:, :, max(0, gz6_nonempty_limit - z0) :]))
                if z1 > gz6_nonempty_limit
                else 0
            )
            if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
                z_sk0 = max(0, eff_gz6_sky_limit - z0)
                sk_sl = new_m[:, :, z_sk0:]
                sk_tot = int(np.count_nonzero(sk_sl))
                if sk_tot >= 8 and int(np.count_nonzero(sk_sl & tgt_sl_nf[:, :, z_sk0:])) * 3 < sk_tot:
                    return -1.0

        is_tp = pid_f in thin_panel_pids
        if enforce_min_precision and ((d_in * 3 < d_ov) if (is_tp or is_thin_regime) else (d_in * 2 < d_ov)):
            return -1.0
        if z1 > gz6_nonempty_limit and sky_ov_f * 2 > d_in:
            return -1.0

        # Guard against low-precision boulders during initial seeding and on thin wireframes
        if enforce_min_precision and sess.inter_live < int(0.18 * mesh_sub_total) and float(d_in) <= 0.22 * float(d_ov):
            return -1.0
        is_sideways_thin_skin = (
            min(x1 - x0, y1 - y0) <= 3 and (z1 - z0) >= 3
        )
        if enforce_min_precision and is_thin_regime and not is_tp and not is_sideways_thin_skin:
            if tot_new >= 48 and float(d_in) < 0.85 * float(d_ov):
                return -1.0
            if tot_new >= 32 and float(d_in) < 0.55 * float(d_ov):
                return -1.0

        u_cur = float(max(1, mesh_sub_total + sess.over_live))
        iou_cur = float(sess.inter_live) / u_cur
        iou_new = float(sess.inter_live + d_in) / float(max(1, mesh_sub_total + sess.over_live + d_ov))
        return iou_new - iou_cur

    def _eval_boxes_delta_iou(
        wbs: Sequence[LDUBox],
        enforce_min_precision: bool = True,
    ) -> Tuple[float, int, int]:
        """Compute exact global `ΔIoU` on `occ6` in ~4 microseconds without mutating `occ6_live`."""
        if mesh_sub_total == 0:
            return -1.0, 0, 0
        sess.candidate_poses_scored += 1
        gx, gy, gz = occ6.shape
        xmin = min(b.xmin for b in wbs)
        xmax = max(b.xmax for b in wbs)
        ymin = min(b.ymin for b in wbs)
        ymax = max(b.ymax for b in wbs)
        zmin = min(b.zmin for b in wbs)
        zmax = max(b.zmax for b in wbs)
        ux0 = int(math.floor(xmin / 5.0))
        ux1 = int(math.ceil(xmax / 5.0))
        uy0 = int(math.floor(ymin / 5.0))
        uy1 = int(math.ceil(ymax / 5.0))
        uz0 = int(math.floor(zmin / 6.0))
        uz1 = int(math.ceil(zmax / 6.0))
        if ux0 < 0 or uy0 < 0 or uz0 < 0 or ux1 > gx or uy1 > gy or uz1 > gz:
            return -1.0, 0, 0
        x0, x1, y0, y1, z0, z1 = ux0, ux1, uy0, uy1, uz0, uz1
        if x1 <= x0 or y1 <= y0 or z1 <= z0 or z0 >= gz6_nonempty_limit:
            return -1.0, 0, 0

        if len(wbs) == 1:
            new_m = ~sess.occ6_live[x0:x1, y0:y1, z0:z1]
            if not np.any(new_m):
                return 0.0, 0, 0
            tgt_sl = occ6[x0:x1, y0:y1, z0:z1]
            d_in = int(np.count_nonzero(new_m & tgt_sl))
            d_ov = int(np.count_nonzero(new_m & (~tgt_sl)))
        else:
            loc = np.zeros((x1 - x0, y1 - y0, z1 - z0), dtype=bool)
            for b in wbs:
                bx0 = max(x0, int(math.floor(b.xmin / 5.0))) - x0
                bx1 = min(x1, int(math.ceil(b.xmax / 5.0))) - x0
                by0 = max(y0, int(math.floor(b.ymin / 5.0))) - y0
                by1 = min(y1, int(math.ceil(b.ymax / 5.0))) - y0
                bz0 = max(z0, int(math.floor(b.zmin / 6.0))) - z0
                bz1 = min(z1, int(math.ceil(b.zmax / 6.0))) - z0
                if bx1 > bx0 and by1 > by0 and bz1 > bz0:
                    loc[bx0:bx1, by0:by1, bz0:bz1] = True
            new_m = loc & (~sess.occ6_live[x0:x1, y0:y1, z0:z1])
            if not np.any(new_m):
                return 0.0, 0, 0
            tgt_sl = occ6[x0:x1, y0:y1, z0:z1]
            d_in = int(np.count_nonzero(new_m & tgt_sl))
            d_ov = int(np.count_nonzero(new_m & (~tgt_sl)))

        if enforce_min_precision and ((d_in * 3 < d_ov) if is_thin_regime else (d_in * 2 < d_ov)):
            return -1.0, d_in, d_ov
        if z1 > gz6_nonempty_limit:
            sky_ov = int(np.count_nonzero(new_m[:, :, max(0, gz6_nonempty_limit - z0) :]))
            if sky_ov * 2 > d_in:
                return -1.0, d_in, d_ov
        if not is_thin_regime and (z1 - z0) >= 3 and z1 > eff_gz6_sky_limit:
            z_sk0 = max(0, eff_gz6_sky_limit - z0)
            sk_sl = new_m[:, :, z_sk0:]
            sk_tot = int(np.count_nonzero(sk_sl))
            if sk_tot >= 8 and int(np.count_nonzero(sk_sl & tgt_sl[:, :, z_sk0:])) * 3 < sk_tot:
                return -1.0, d_in, d_ov

        if (
            enforce_min_precision
            and sess.inter_live < int(0.18 * mesh_sub_total)
            and float(d_in) <= 0.22 * float(d_ov)
        ):
            return -1.0, d_in, d_ov

        u_cur = float(max(1, mesh_sub_total + sess.over_live))
        if view_weight_grid_6 is not None:
            w_sl = view_weight_grid_6[x0:x1, y0:y1, z0:z1]
            d_in_w = float(np.sum(w_sl[new_m & tgt_sl]))
            d_ov_w = float(np.sum(w_sl[new_m & (~tgt_sl)]))
            if enforce_min_precision and d_in_w * 1.65 < d_ov_w:
                return -1.0, d_in, d_ov
            iou_cur_w = float(sess.inter_live) / u_cur
            iou_new_w = (float(sess.inter_live) + d_in_w) / float(max(1.0, u_cur + d_ov_w))
            return (iou_new_w - iou_cur_w), d_in, d_ov
        iou_cur = float(sess.inter_live) / u_cur
        iou_new = float(sess.inter_live + d_in) / float(max(1, mesh_sub_total + sess.over_live + d_ov))
        return (iou_new - iou_cur), d_in, d_ov

    def _protrusion_clear_against_existing(pid_c: str, pose_c: LDUFrame) -> bool:
        """O(1) spatial-hash check that no unmated stud protrusion clashes with existing protrusions."""
        prim_p = get_brick_primitive(pid_c)
        for p_spec in prim_p.ports:
            if p_spec.local_protrusion_box is not None:
                w_pf = pose_c.compose(p_spec.local_frame)
                w_pr = p_spec.local_protrusion_box.transform(w_pf)
                for cell in SpatialBoxHashIndex.box_cells(w_pr):
                    for _, other_pr in sess.state._protrusion_cells.get(cell, ()):
                        if w_pr.intersects(other_pr):
                            return False
        return True

    sess.skin_lp_calls = 0
    max_skin_lp_calls = 36 if is_thin_regime else 22
    sess.consec_skin_lp_fails = 0
    sess.deferred_lp_in_batch = False
    sess.batch_new_commits: List[Tuple[str, LDUFrame, bool]] = []

    def _deadline_expired() -> bool:
        return deadline_monotonic_s is not None and time.monotonic() >= deadline_monotonic_s

    def _try_commit_part(
        pid_c: str,
        pose_c: LDUFrame,
        grounded_c: bool,
        check_csg_strict: bool = False,
        budget_lp: bool = False,
        defer_lp: bool = False,
        plumb_only: bool = False,
    ) -> bool:
        if _deadline_expired():
            return False
        if sess.rem_inv is not None and sess.rem_inv.get(pid_c, 0) <= 0:
            return False
        prim_c = get_brick_primitive(pid_c)
        wbs = tuple(b.transform(pose_c) for b in prim_c.hull_boxes)
        if any(_box_hits_forbidden(wb, forbidden_support_boxes) for wb in wbs):
            return False
        if not check_monotone_insertion_corridor(sess.state, pid_c, pose_c):
            return False
        if check_csg_strict and not _protrusion_clear_against_existing(pid_c, pose_c):
            return False

        tok = None
        try:
            tok = sess.state.push_trial_part(pid_c, pose_c, is_grounded=grounded_c)
            if not grounded_c and len(tok.added_adj_other_insts) == 0:
                sess.state.pop_trial_part(tok)
                return False
            trial_iid = tok.inst_id
            stable_ok = False
            if (
                grounded_c
                and sess.base_lp_eta <= 1.0 - 1e-5
                and (
                    foundation_policy == FoundationPolicy.CLUTCHED_BASEPLATE
                    or (
                        foundation_policy == FoundationPolicy.TABLE_FRICTION_ONLY
                        and wbs
                        and any(
                            abs(float(b.zmin) - ground_plane_z_ldu) <= 0.25
                            and (b.xmin + 0.5)
                            <= (
                                sum(float(bx.volume_ldu3) * 0.5 * (bx.xmin + bx.xmax) for bx in wbs)
                                / max(1.0, sum(float(bx.volume_ldu3) for bx in wbs))
                            )
                            <= (b.xmax - 0.5)
                            and (b.ymin + 0.5)
                            <= (
                                sum(float(bx.volume_ldu3) * 0.5 * (bx.ymin + bx.ymax) for bx in wbs)
                                / max(1.0, sum(float(bx.volume_ldu3) for bx in wbs))
                            )
                            <= (b.ymax - 0.5)
                            for b in wbs
                        )
                    )
                )
            ):
                stable_ok = True
            elif (
                foundation_policy in (FoundationPolicy.CLUTCHED_BASEPLATE, FoundationPolicy.TABLE_FRICTION_ONLY)
                and sess.base_lp_eta <= 1.0 - 1e-5
            ):
                can_ext, peak_d = kernel.validator.can_extend_plumb_primal_witness(
                    sess.state,
                    trial_iid,
                    sess.plumb_deltas,
                    base_witness=sess.base_witness,
                    hop_cache=sess.plumb_hop_cache,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                )
                if not can_ext and sess.state.num_parts >= 40 and not defer_lp:
                    can_ext, peak_d = kernel.validator.verify_two_pier_bearing_compression(
                        sess.state,
                        foundation_policy=foundation_policy,
                        ground_plane_z_ldu=ground_plane_z_ldu,
                    )
                if can_ext and max(sess.base_lp_eta, peak_d) <= 1.0 - 1e-5:
                    stable_ok = True
            if (
                not stable_ok
                and defer_lp
                and foundation_policy in (FoundationPolicy.CLUTCHED_BASEPLATE, FoundationPolicy.TABLE_FRICTION_ONLY)
                and (sess.base_lp_eta <= 0.82 or is_megastructure_footprint)
                and (prim_c.mass_grams <= 8.0 or is_megastructure_footprint)
            ):
                stable_ok = True
                sess.deferred_lp_in_batch = True
            if not stable_ok:
                if plumb_only:
                    sess.plumb_hop_cache.pop(trial_iid, None)
                    sess.state.pop_trial_part(tok)
                    return False
                if budget_lp and (sess.skin_lp_calls >= max_skin_lp_calls or sess.consec_skin_lp_fails >= 10):
                    sess.plumb_hop_cache.pop(trial_iid, None)
                    sess.state.pop_trial_part(tok)
                    return False
                if budget_lp:
                    sess.skin_lp_calls += 1
                eq_res = kernel.validator.evaluate(
                    sess.state,
                    foundation_policy=foundation_policy,
                    ground_plane_z_ldu=ground_plane_z_ldu,
                    deadline_monotonic_s=deadline_monotonic_s,
                )
                if not eq_res.is_stable:
                    if budget_lp:
                        sess.consec_skin_lp_fails += 1
                    sess.plumb_hop_cache.pop(trial_iid, None)
                    sess.state.pop_trial_part(tok)
                    return False
                if budget_lp:
                    sess.consec_skin_lp_fails = 0
                sess.base_witness = eq_res.primal_witness
                sess.base_lp_eta = float(eq_res.peak_utilization)
                sess.plumb_deltas.clear()
            if sess.plumb_hop_cache:
                w_zmax = max(b.zmax for b in wbs)
                for adj_id in tok.added_adj_other_insts:
                    if adj_id in sess.plumb_hop_cache and min(b.zmin for b in sess.state._parts[adj_id].world_boxes) >= w_zmax:
                        sess.plumb_hop_cache.pop(adj_id, None)
            sess.state.pop_trial_part(tok)
            tok = None

            sess.state.place_root_part(pid_c, pose_c, is_grounded=grounded_c)
            sess.build_mold_counts[pid_c] = sess.build_mold_counts.get(pid_c, 0) + 1
            if sess.rem_inv is not None:
                sess.rem_inv[pid_c] = sess.rem_inv.get(pid_c, 0) - 1
            _add_boxes_to_live_occ6(wbs)
            if defer_lp:
                sess.batch_new_commits.append((pid_c, pose_c, grounded_c))
            return True
        except ValueError:
            if tok is not None:
                sess.plumb_hop_cache.pop(tok.inst_id, None)
                sess.state.pop_trial_part(tok)
            return False

    is_megastructure_footprint = (w_x * w_y) >= 169

    # Execute planned bottom-up core & jumper layers; queue cantilevered spans that need neighbor
    # bridging within the same Z layer for a second pass right after pier anchors commit!
    if _deadline_expired():
        planned_by_z.clear()
    for z0 in sorted(planned_by_z.keys()):
        if _deadline_expired():
            break
        layer_items = planned_by_z[z0]
        is_gr = abs(float(z0) - ground_plane_z_ldu) <= 0.25
        pending = list(layer_items)
        use_course_defer = is_megastructure_footprint and not is_gr and not is_hollow_shell_active
        if use_course_defer:
            g_xs_oc = [float(b.xmin) for p in sess.state.placed_parts.values() if p.is_grounded for b in p.world_boxes] + [
                float(b.xmax) for p in sess.state.placed_parts.values() if p.is_grounded for b in p.world_boxes
            ]
            g_ys_oc = [float(b.ymin) for p in sess.state.placed_parts.values() if p.is_grounded for b in p.world_boxes] + [
                float(b.ymax) for p in sess.state.placed_parts.values() if p.is_grounded for b in p.world_boxes
            ]
            gx_c = 0.5 * (min(g_xs_oc) + max(g_xs_oc)) if g_xs_oc else 10.0 * float(w_x)
            gy_c = 0.5 * (min(g_ys_oc) + max(g_ys_oc)) if g_ys_oc else 10.0 * float(w_y)
            pending.sort(key=lambda it: (float(it[1].pos[0]) - gx_c) ** 2 + (float(it[1].pos[1]) - gy_c) ** 2)
            # Phase 1: Permanently commit all plumb-supported core bricks in O(1) with ZERO LP solves!
            for _pass_pl in range(3):
                if not pending or _deadline_expired():
                    break
                still_pl: List[Tuple[str, LDUFrame]] = []
                prog_pl = False
                for pid_l, pose_l in pending:
                    if _deadline_expired():
                        still_pl.append((pid_l, pose_l))
                        continue
                    if _try_commit_part(
                        pid_l, pose_l, grounded_c=False, check_csg_strict=False, budget_lp=False, defer_lp=False, plumb_only=True
                    ):
                        prog_pl = True
                    else:
                        still_pl.append((pid_l, pose_l))
                pending = still_pl
                if not prog_pl:
                    break
            # Phase 2: Remaining overhanging/cantilevered bricks sorted center-outward; stop outward once clutch saturates!
            if pending and not _deadline_expired():
                cant_lp_fails = 0
                for _pass_c in range(2):
                    if not pending or _deadline_expired() or cant_lp_fails >= 3:
                        break
                    still_c: List[Tuple[str, LDUFrame]] = []
                    prog_c = False
                    for pid_l, pose_l in pending:
                        if _deadline_expired() or cant_lp_fails >= 3:
                            still_c.append((pid_l, pose_l))
                            continue
                        prev_parts_n = sess.state.num_parts
                        if _try_commit_part(
                            pid_l, pose_l, grounded_c=False, check_csg_strict=False, budget_lp=True, defer_lp=False, plumb_only=False
                        ):
                            prog_c = True
                            cant_lp_fails = 0
                        else:
                            still_c.append((pid_l, pose_l))
                            if sess.state.num_parts == prev_parts_n:
                                cant_lp_fails += 1
                    pending = still_c
                    if not prog_c:
                        break
        else:
            for _pass in range(4):
                if not pending or _deadline_expired():
                    break
                still_pending: List[Tuple[str, LDUFrame]] = []
                progressed = False
                for pid_l, pose_l in pending:
                    if _deadline_expired():
                        still_pending.append((pid_l, pose_l))
                        continue
                    if _try_commit_part(
                        pid_l, pose_l, grounded_c=is_gr, check_csg_strict=False, budget_lp=False, defer_lp=False
                    ):
                        progressed = True
                    else:
                        still_pending.append((pid_l, pose_l))
                pending = still_pending
                if not progressed:
                    break
        # Fallback: if any reserved multi-box slope/inverted-slope failed insertion/LP, fill its `core_occ` cells
        # with `1x1` bricks/plates so zero structural holes propagate to upper courses!
        if pending:
            for pid_f, pose_f in pending:
                prim_f = get_brick_primitive(pid_f)
                if len(prim_f.hull_boxes) <= 1:
                    continue
                wbs_f = (
                    [prim_f.hull_boxes[0].transform(pose_f)]
                    if pid_f in ("44728", "99780", "99207")
                    else [b.transform(pose_f) for b in prim_f.hull_boxes]
                )
                f_xmin = min(b.xmin for b in wbs_f)
                f_xmax = max(b.xmax for b in wbs_f)
                f_ymin = min(b.ymin for b in wbs_f)
                f_ymax = max(b.ymax for b in wbs_f)
                f_zmax = max(b.zmax for b in wbs_f)
                p_f0 = z0 // 8
                p_f1 = min(n_plates_z, f_zmax // 8)
                if p_f0 >= n_plates_z:
                    continue
                ph_fb = layer_phase[p_f0]
                rx0_f = max(0, (f_xmin - ph_fb[0]) // 20)
                rx1_f = min(w_x, (f_xmax - ph_fb[0]) // 20)
                ry0_f = max(0, (f_ymin - ph_fb[1]) // 20)
                ry1_f = min(w_y, (f_ymax - ph_fb[1]) // 20)
                if "3024" in kit_set:
                    p1_box = get_brick_primitive("3024").hull_boxes[0]
                    for pq_fb in range(p_f0, p_f1):
                        z_fb = pq_fb * 8
                        for fx_i in range(rx0_f, rx1_f):
                            for fy_i in range(ry0_f, ry1_f):
                                if core_occ_pre_sculpt[fx_i, fy_i, pq_fb]:
                                    pose_p1 = LDUFrame(
                                        0,
                                        (
                                            fx_i * 20 + ph_fb[0] - p1_box.xmin,
                                            fy_i * 20 + ph_fb[1] - p1_box.ymin,
                                            z_fb - p1_box.zmin,
                                        ),
                                    )
                                    _try_commit_part("3024", pose_p1, grounded_c=(z_fb == 0), check_csg_strict=False, budget_lp=True)


    sess.mesh_sub_total = int(mesh_sub_total)
    sess.max_skin_lp_calls = int(max_skin_lp_calls)
    sess.is_megastructure_footprint = bool(is_megastructure_footprint)
    sess.eval_fast_pose_delta_iou = _eval_fast_pose_delta_iou
    sess.eval_boxes_delta_iou = _eval_boxes_delta_iou
    sess.protrusion_clear_against_existing = _protrusion_clear_against_existing
    sess.deadline_expired = _deadline_expired
    sess.try_commit_part = _try_commit_part
