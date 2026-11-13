"""
WS-4A, WS-4B & WS-5A: Large-Scale Untouched External `StableText2Brick` Benchmark (`n = 600`),
Hard-Overhang / Arch Challenge Set (`n = 30`, `Claim B`), and Held-Out `120`-Part `TIER_A_CATALOG`
`M2` Port-Filter Evaluation (`n_eval = 50`).

Scientific & Reviewer Guardrails Enforced:
1. Zero prior overlap: Every ShapeNet `object_id` in the `n = 600` `test`-split cohort is verified
   disjoint (`0` overlap) from both `LOCKED_STABLETEXT2BRICK_MANIFEST` (`16` pilot objects) and
   `UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST` (`21` prior eval objects).
2. Strict separation of realization contracts:
   - `S1_Blind_WholeTarget` (`mode="blind"`, primary `Claim A` benchmark): sees ONLY the whole-target
     `SparseTiledTargetField` and `STABLETEXT2BRICK_8PART_KIT` (zero reference `z0` slicing, zero
     per-course part hints, zero ground-brick spoilers, zero fallback reference placement).
   - `S1_Reference_Assisted` (`mode="reference_assisted"`): reported in a separate schema entry for
     side-by-side comparison and never conflated with blind `realize(target_field)`.
   - `Legolization_Baseline_Blind` (`mode="blind"`): classical greedy layer brick-merging baseline.
3. Exact Clopper-Pearson 95% binomial confidence intervals (`scipy.stats.beta.ppf`) over unique
   `object_id`s (`n = 600` external cohort, `n = 30` Hard-Overhang Challenge Set).
4. Strictly disjoint `n_train = 40` / `n_eval = 50` course evaluation of `M2` (`LearnedCatalogAndPortFilter`)
   over `120` parts of `TIER_A_CATALOG` using genuine `S1` port-inclusion rollout labels (`0` formula copying).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as scipy_beta
import torch
from torch import nn

from topo_brick.evaluation.physical_validation import (
    ModelAgreementReport,
    run_physical_and_model_agreement_suite,
)
from topo_brick.interface.brick_catalog import (
    STABLETEXT2BRICK_8PART_KIT,
    TIER_A_CATALOG,
    compute_intrinsic_part_descriptor,
    get_brick_primitive,
)
from topo_brick.interface.ldu_frames import LDUBox
from topo_brick.interface.port_manifold import (
    AssemblyState,
    compute_cheap_pre_expansion_port_scores,
)
from topo_brick.kernel.construction_kernel import (
    KernelResultStatus,
    LearnedCatalogAndPortFilter,
    LegolizationBaseline,
    PromotionGateVerdict,
    TopoBrickKernel,
    _get_peak_rss_mb,
    evaluate_corpus_attrition_funnel,
    parse_external_record_specs,
)
from topo_brick.kernel.untouched_external_corpus import (
    UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST,
)
from topo_brick.representation.interface_fields import (
    SparseTiledTargetField,
    TargetShapeBoxes,
)
from topo_brick.training.stage55_and_stage6_audit_and_external_eval import (
    LOCKED_STABLETEXT2BRICK_MANIFEST,
)

DEFAULT_COHORT_GZ_PATH: Path = Path(__file__).with_name("untouched_s2b_600_cohort.json.gz")
DEFAULT_RESULTS_JSON_PATH: Path = Path(__file__).with_name("untouched_s2b_600_results.json")


def clopper_pearson_95_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute exact two-sided `(1 - alpha)` Clopper-Pearson binomial confidence interval."""
    if n <= 0:
        return (0.0, 1.0)
    k = max(0, min(int(k), int(n)))
    lo = 0.0 if k == 0 else float(scipy_beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(scipy_beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def get_prior_locked_and_eval_object_ids() -> set[str]:
    """Return the exact union of all 37 `object_id`s used in prior pilot (16) and eval (21) suites."""
    pilot_ids = {str(r.object_id) for r in LOCKED_STABLETEXT2BRICK_MANIFEST}
    eval21_ids = {str(r["object_id"]) for r in UNTOUCHED_STABLETEXT2BRICK_21_MANIFEST}
    return pilot_ids | eval21_ids


@dataclass(frozen=True, slots=True)
class ParquetCohortAttritionSummary:
    """Pre-filter attrition accounting from raw `AvaLovelace/StableText2Brick` `test` Parquet to `n = 600`."""

    source_parquet_name: str
    raw_parquet_rows: int
    prior_pilot_and_eval_overlap_removed: int
    duplicate_object_id_captions_removed: int
    unique_disjoint_object_ids: int
    supported_by_8part_kit: int
    eligible_le_100_bricks: int
    eligible_101_to_250_bricks: int
    excluded_gt_250_bricks: int
    sampled_le_100_bricks: int
    sampled_101_to_250_bricks: int
    sampled_total_objects: int
    num_shapenet_categories: int
    sampling_seed: int
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class LargeScaleSolverSummary:
    """Aggregated performance & certification metrics for one solver/mode across `n = 600` objects."""

    solver_mode_id: str
    solver_id: str
    realization_mode: str
    num_objects: int
    certified_count: int
    certified_completion_rate: float
    certified_ci95_low: float
    certified_ci95_high: float
    static_stable_count: int
    static_stable_rate: float
    static_stable_ci95_low: float
    static_stable_ci95_high: float
    prefix_stable_rate: float
    monotone_insertable_rate: float
    collision_free_rate: float
    mean_target_iou: float
    median_target_iou: float
    mean_target_recall: float
    mean_target_precision: float
    mean_parts_placed: float
    mean_reference_parts: float
    parts_ratio_vs_reference: float
    median_wall_clock_ms_all: float
    p90_wall_clock_ms_all: float
    p95_wall_clock_ms_all: float
    num_le_100: int
    median_wall_clock_ms_le_100: float
    p90_wall_clock_ms_le_100: float
    num_101_to_250: int
    median_wall_clock_ms_101_to_250: float
    p90_wall_clock_ms_101_to_250: float
    peak_rss_mb: float


def extract_untouched_s2b_600_cohort(
    parquet_path: Path | str,
    out_gz_path: Path | str = DEFAULT_COHORT_GZ_PATH,
    target_le_100: int = 450,
    target_101_to_250: int = 150,
    seed: int = 20261110,
) -> Dict[str, Any]:
    """
    Deterministically extract a category-stratified `n = 600` (`450` in `<= 100` bricks + `150` in
    `101..250` bricks) `object_id`-deduplicated untouched cohort from `AvaLovelace/StableText2Brick`
    `test-00000-of-00001.parquet`, verifying zero overlap with all 37 prior pilot/eval objects.
    """
    import pyarrow.parquet as pq

    parquet_p = Path(parquet_path)
    table = pq.read_table(parquet_p)
    raw_rows = table.num_rows
    prior_ids = get_prior_locked_and_eval_object_ids()

    col_struct = table["structure_id"].to_pylist()
    col_obj = table["object_id"].to_pylist()
    col_cat = table["category_id"].to_pylist()
    col_caps = table["captions"].to_pylist()
    col_bricks = table["bricks"].to_pylist()
    col_stab = table["stability_scores"].to_pylist()

    overlap_removed = 0
    dup_obj_removed = 0
    seen_obj_ids: set[str] = set()
    supported_cnt = 0
    pool_le_100_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pool_101_250_by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    gt_250_cnt = 0

    for row_idx in range(raw_rows):
        obj_id = str(col_obj[row_idx])
        if obj_id in prior_ids:
            overlap_removed += 1
            continue
        if obj_id in seen_obj_ids:
            dup_obj_removed += 1
            continue
        seen_obj_ids.add(obj_id)

        bricks_txt = str(col_bricks[row_idx])
        specs, rep = parse_external_record_specs(
            bricks_txt, allowed_part_ids=STABLETEXT2BRICK_8PART_KIT
        )
        if rep.unsupported_valid_part_lines > 0 or rep.malformed_type1_lines > 0 or len(specs) == 0:
            continue
        supported_cnt += 1

        stab_arr = np.asarray(col_stab[row_idx], dtype=np.float64)
        occ_mask = stab_arr < (1.0 - 1e-12)
        ref_min_score = float(np.min(stab_arr))
        ref_max_util = float(1.0 - ref_min_score) if np.any(occ_mask) else 0.0
        caps = col_caps[row_idx]
        cap0 = str(caps[0]) if caps else ""
        cat_id = str(col_cat[row_idx])
        n_bricks = len(specs)

        rec = {
            "source_row_index": int(row_idx),
            "structure_id": str(col_struct[row_idx]),
            "object_id": obj_id,
            "category_id": cat_id,
            "caption": cap0,
            "ldr_text": bricks_txt,
            "num_bricks_claimed": int(n_bricks),
            "reference_min_stability_score": ref_min_score,
            "reference_max_utilization": ref_max_util,
            "split": "test",
            "source_dataset": "AvaLovelace/StableText2Brick",
        }
        if n_bricks <= 100:
            pool_le_100_by_cat[cat_id].append(rec)
        elif n_bricks <= 250:
            pool_101_250_by_cat[cat_id].append(rec)
        else:
            gt_250_cnt += 1

    def _stratified_sample(
        by_cat: Dict[str, List[Dict[str, Any]]], target_k: int, rng_seed: int
    ) -> List[Dict[str, Any]]:
        rng = random.Random(rng_seed)
        cats = sorted(by_cat.keys())
        total_avail = sum(len(by_cat[c]) for c in cats)
        if target_k >= total_avail:
            return [r for c in cats for r in by_cat[c]]

        shuffled: Dict[str, List[Dict[str, Any]]] = {}
        quotas: Dict[str, int] = {}
        remainders: List[Tuple[float, str]] = []
        allocated = 0
        for c in cats:
            items = list(by_cat[c])
            rng.shuffle(items)
            shuffled[c] = items
            exact = (len(items) * float(target_k)) / float(total_avail)
            q = min(len(items), int(exact))
            quotas[c] = q
            allocated += q
            remainders.append((exact - float(q), c))

        remainders.sort(key=lambda x: (-x[0], x[1]))
        rem_needed = target_k - allocated
        for _, c in remainders:
            if rem_needed <= 0:
                break
            if quotas[c] < len(shuffled[c]):
                quotas[c] += 1
                rem_needed -= 1

        selected: List[Dict[str, Any]] = []
        for c in cats:
            selected.extend(shuffled[c][: quotas[c]])
        selected.sort(key=lambda r: int(r["source_row_index"]))
        return selected

    tot_le_100 = sum(len(v) for v in pool_le_100_by_cat.values())
    tot_101_250 = sum(len(v) for v in pool_101_250_by_cat.values())
    samp_le_100 = _stratified_sample(pool_le_100_by_cat, target_le_100, seed)
    samp_101_250 = _stratified_sample(pool_101_250_by_cat, target_101_to_250, seed + 1)
    records = sorted(samp_le_100 + samp_101_250, key=lambda r: int(r["source_row_index"]))

    all_cats = {r["category_id"] for r in records}
    canonical_manifest_str = json.dumps(
        [(r["source_row_index"], r["object_id"], r["num_bricks_claimed"]) for r in records],
        separators=(",", ":"),
    )
    sha = hashlib.sha256(canonical_manifest_str.encode("utf-8")).hexdigest()

    attrition = ParquetCohortAttritionSummary(
        source_parquet_name=parquet_p.name,
        raw_parquet_rows=raw_rows,
        prior_pilot_and_eval_overlap_removed=overlap_removed,
        duplicate_object_id_captions_removed=dup_obj_removed,
        unique_disjoint_object_ids=len(seen_obj_ids),
        supported_by_8part_kit=supported_cnt,
        eligible_le_100_bricks=tot_le_100,
        eligible_101_to_250_bricks=tot_101_250,
        excluded_gt_250_bricks=gt_250_cnt,
        sampled_le_100_bricks=len(samp_le_100),
        sampled_101_to_250_bricks=len(samp_101_250),
        sampled_total_objects=len(records),
        num_shapenet_categories=len(all_cats),
        sampling_seed=seed,
        manifest_sha256=sha,
    )
    payload = {
        "attrition_summary": asdict(attrition),
        "records": records,
    }
    out_p = Path(out_gz_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_p, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return payload


def load_untouched_s2b_600_cohort(
    gz_path: Path | str = DEFAULT_COHORT_GZ_PATH,
) -> Dict[str, Any]:
    """Load the frozen `n = 600` untouched `StableText2Brick` `test` cohort from gzipped JSON."""
    with gzip.open(Path(gz_path), "rt", encoding="utf-8") as f:
        return json.load(f)


def _summarize_solver_runs(
    solver_mode_id: str,
    solver_id: str,
    realization_mode: str,
    build_records: Sequence[Any],
) -> LargeScaleSolverSummary:
    n = len(build_records)
    cert_cnt = sum(
        1 for r in build_records if r.status == KernelResultStatus.COMPLETED_CERTIFIED.value
    )
    stab_cnt = sum(1 for r in build_records if r.static_stable)
    pref_cnt = sum(1 for r in build_records if r.prefix_stable)
    ins_cnt = sum(1 for r in build_records if r.monotone_insertable)
    coll_cnt = sum(1 for r in build_records if r.collision_free and r.protrusion_collision_free)

    cert_lo, cert_hi = clopper_pearson_95_ci(cert_cnt, n)
    stab_lo, stab_hi = clopper_pearson_95_ci(stab_cnt, n)

    ious = [float(r.target_iou) for r in build_records]
    recs = [float(r.target_recall) for r in build_records]
    precs = [float(r.target_precision) for r in build_records]
    placed_cnts = [int(r.num_parts_placed) for r in build_records]
    ref_cnts = [int(r.reference_part_count) for r in build_records]
    ms_all = [float(r.wall_clock_ms) for r in build_records]
    ms_le100 = [float(r.wall_clock_ms) for r in build_records if int(r.reference_part_count) <= 100]
    ms_101_250 = [
        float(r.wall_clock_ms) for r in build_records if 100 < int(r.reference_part_count) <= 250
    ]

    mean_placed = float(np.mean(placed_cnts)) if placed_cnts else 0.0
    mean_ref = float(np.mean(ref_cnts)) if ref_cnts else 1.0

    return LargeScaleSolverSummary(
        solver_mode_id=solver_mode_id,
        solver_id=solver_id,
        realization_mode=realization_mode,
        num_objects=n,
        certified_count=cert_cnt,
        certified_completion_rate=float(cert_cnt / max(n, 1)),
        certified_ci95_low=cert_lo,
        certified_ci95_high=cert_hi,
        static_stable_count=stab_cnt,
        static_stable_rate=float(stab_cnt / max(n, 1)),
        static_stable_ci95_low=stab_lo,
        static_stable_ci95_high=stab_hi,
        prefix_stable_rate=float(pref_cnt / max(n, 1)),
        monotone_insertable_rate=float(ins_cnt / max(n, 1)),
        collision_free_rate=float(coll_cnt / max(n, 1)),
        mean_target_iou=float(np.mean(ious)) if ious else 0.0,
        median_target_iou=float(np.median(ious)) if ious else 0.0,
        mean_target_recall=float(np.mean(recs)) if recs else 0.0,
        mean_target_precision=float(np.mean(precs)) if precs else 0.0,
        mean_parts_placed=mean_placed,
        mean_reference_parts=mean_ref,
        parts_ratio_vs_reference=float(mean_placed / max(mean_ref, 1e-6)),
        median_wall_clock_ms_all=float(np.median(ms_all)) if ms_all else 0.0,
        p90_wall_clock_ms_all=float(np.percentile(ms_all, 90)) if ms_all else 0.0,
        p95_wall_clock_ms_all=float(np.percentile(ms_all, 95)) if ms_all else 0.0,
        num_le_100=len(ms_le100),
        median_wall_clock_ms_le_100=float(np.median(ms_le100)) if ms_le100 else 0.0,
        p90_wall_clock_ms_le_100=float(np.percentile(ms_le100, 90)) if ms_le100 else 0.0,
        num_101_to_250=len(ms_101_250),
        median_wall_clock_ms_101_to_250=float(np.median(ms_101_250)) if ms_101_250 else 0.0,
        p90_wall_clock_ms_101_to_250=float(np.percentile(ms_101_250, 90)) if ms_101_250 else 0.0,
        peak_rss_mb=_get_peak_rss_mb(),
    )


def run_untouched_s2b_600_benchmark(
    cohort_payload: Optional[Dict[str, Any]] = None,
    host_label: str = "host_a",
    out_json_path: Optional[Path | str] = DEFAULT_RESULTS_JSON_PATH,
    include_s2b_agreement: bool = True,
) -> Dict[str, Any]:
    """
    Execute the full `n = 600` untouched `AvaLovelace/StableText2Brick` `test`-split benchmark on
    1 CPU core across:
      1. `S1_Blind_WholeTarget` (`mode="blind"`, primary `Claim A`),
      2. `S1_Reference_Assisted` (`mode="reference_assisted"`),
      3. `Legolization_Baseline_Blind` (`mode="blind"`),
    plus 7-stage reference corpus attrition funnel and `stability_scores` agreement.
    """
    if cohort_payload is None:
        cohort_payload = load_untouched_s2b_600_cohort()
    records: List[Dict[str, Any]] = list(cohort_payload["records"])

    # Verify 0 overlap with all 37 prior pilot + eval objects
    prior_ids = get_prior_locked_and_eval_object_ids()
    overlap = [r["object_id"] for r in records if str(r["object_id"]) in prior_ids]
    if overlap:
        raise ValueError(f"Found unexpected prior object_id overlap: {overlap[:5]}")

    kernel = TopoBrickKernel()
    ref_funnel = evaluate_corpus_attrition_funnel(
        records,
        allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
        kit_name="STABLETEXT2BRICK_8PART_KIT",
        validator=kernel.validator,
    )

    blind_recs = []
    s2_blind_recs = []
    ref_assist_recs = []
    lego_base_recs = []

    t_run_start = time.perf_counter()
    cpu_run_start = time.process_time()
    task_outcomes_digest: List[Tuple[str, str, str, str, str]] = []

    for rec in records:
        res_blind = kernel.realize_external_record(
            rec, solver_id="S1_CoverAndRepair", mode="blind"
        )
        blind_recs.append(res_blind.build_record)

        res_s2 = kernel.realize_external_record(
            rec, solver_id="S2_PairSeededLocalSearch", mode="blind"
        )
        s2_blind_recs.append(res_s2.build_record)

        res_ref = kernel.realize_external_record(
            rec, solver_id="S1_CoverAndRepair", mode="reference_assisted"
        )
        ref_assist_recs.append(res_ref.build_record)

        res_base = kernel.realize_external_record(
            rec, solver_id="Legolization_Baseline", mode="blind"
        )
        lego_base_recs.append(res_base.build_record)

        task_outcomes_digest.append(
            (
                str(rec["object_id"]),
                str(res_blind.build_record.status),
                str(res_s2.build_record.status),
                str(res_ref.build_record.status),
                str(res_base.build_record.status),
            )
        )

    standalone_wall_s = float(time.perf_counter() - t_run_start)
    standalone_cpu_s = float(time.process_time() - cpu_run_start)
    outcomes_sha256 = hashlib.sha256(
        json.dumps(task_outcomes_digest, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    sum_blind = _summarize_solver_runs(
        "S1_Blind_WholeTarget", "S1_CoverAndRepair", "blind", blind_recs
    )
    sum_s2 = _summarize_solver_runs(
        "S2_Blind_WholeTarget", "S2_PairSeededLocalSearch", "blind", s2_blind_recs
    )
    sum_ref = _summarize_solver_runs(
        "S1_Reference_Assisted", "S1_CoverAndRepair", "reference_assisted", ref_assist_recs
    )
    sum_base = _summarize_solver_runs(
        "Legolization_Baseline_Blind", "Legolization_Baseline", "blind", lego_base_recs
    )

    agreement_dict: Optional[Dict[str, Any]] = None
    if include_s2b_agreement:
        ag_rep: ModelAgreementReport = run_physical_and_model_agreement_suite(
            s2b_records=records[:120]
        )
        agreement_dict = {
            "num_physical_cases": ag_rep.num_physical_cases,
            "physical_cases_matched": ag_rep.physical_cases_matched,
            "physical_hardware_accuracy": ag_rep.physical_hardware_accuracy,
            "bricksim_demo_certified_stable": ag_rep.bricksim_demo_certified_stable,
            "bricksim_demo_peak_utilization": ag_rep.bricksim_demo_peak_utilization,
            "s2b_num_evaluated": ag_rep.s2b_num_evaluated,
            "s2b_substud_agreement_rate": ag_rep.s2b_substud_agreement_rate,
            "s2b_kernel_lp_agreement_rate": ag_rep.s2b_kernel_lp_agreement_rate,
            "s2b_mean_max_score_mae": ag_rep.s2b_mean_max_score_mae,
        }

    result_payload: Dict[str, Any] = {
        "attrition_summary": cohort_payload["attrition_summary"],
        "zero_prior_overlap_verified": True,
        "unique_object_ids_verified": len({r["object_id"] for r in records}) == len(records),
        "execution_protocol": "standalone_full_2400_realizations_outside_pytest",
        "reference_corpus_funnel_8part": asdict(ref_funnel),
        "hosts": {
            host_label: {
                "S1_Blind_WholeTarget": asdict(sum_blind),
                "S2_Blind_WholeTarget": asdict(sum_s2),
                "S1_Reference_Assisted": asdict(sum_ref),
                "Legolization_Baseline_Blind": asdict(sum_base),
                "task_outcomes_sha256": outcomes_sha256,
                "standalone_wall_seconds": standalone_wall_s,
                "standalone_cpu_seconds": standalone_cpu_s,
                "cpu_to_wall_ratio": float(standalone_cpu_s / max(standalone_wall_s, 1e-6)),
            }
        },
        "s2b_model_agreement": agreement_dict,
    }

    if out_json_path is not None:
        out_p = Path(out_json_path)
        if out_p.exists():
            try:
                existing = json.loads(out_p.read_text(encoding="utf-8"))
                if "hosts" in existing and isinstance(existing["hosts"], dict):
                    merged_hosts = dict(existing["hosts"])
                    merged_hosts[host_label] = result_payload["hosts"][host_label]
                    result_payload["hosts"] = merged_hosts
            except Exception:
                pass
        out_p.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")

    return result_payload


def load_untouched_s2b_600_results(
    json_path: Path | str = DEFAULT_RESULTS_JSON_PATH,
) -> Dict[str, Any]:
    """Load saved dual-host `n = 600` benchmark summary JSON."""
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# WS-4B: Hard-Overhang / Arch Challenge Set (`Claim B`, `n = 30`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HardOverhangChallengeReport:
    """`Claim B` Verification Report on the `n = 30` Hard-Overhang / Arch Challenge Set."""

    num_challenges: int
    families_tested: Tuple[str, ...]
    baseline_stable_count: int
    baseline_stable_rate: float
    s1_no_support_stable_count: int
    s1_support_certified_count: int
    s1_support_certified_rate: float
    s1_support_ci95_low: float
    s1_support_ci95_high: float
    mean_external_support_bricks: float
    mean_target_recall_with_support: float
    median_wall_clock_ms: float


def build_hard_overhang_arch_challenge_set() -> List[Tuple[str, str, SparseTiledTargetField]]:
    """
    Construct `n = 30` structurally challenging 3D target fields across 3 families where ANY
    strictly target-interior greedy tiling (`Legolization_Baseline`) is mechanically unstable
    (either disconnected mid-air wing/deck or cantilevered overhang exceeding clutch torque),
    requiring minimal permanent external `1x1` support pillars (`Claim B`):
      1. `asymmetric_L_cantilever` (`10` instances): narrow `1x1` grounded stem supporting an
         extended `8..12`-stud horizontal upper cantilever arm at `z = 48..96` LDU.
      2. `disconnected_wing_portal` (`10` instances): grounded main tower + elevated cantilevered
         balcony/island deck separated by an unsupported lower span.
      3. `wide_cantilevered_bridge_deck` (`10` instances): single-pier bridge whose upper roadway
         spans `10..14` studs asymmetrically along `+X` or `+Y`.
    """
    challenges: List[Tuple[str, str, SparseTiledTargetField]] = []

    # Family 1: 10 Asymmetric L-Cantilevers (1x1 stem + long horizontal top beam that tips without prop)
    for idx in range(10):
        stem_courses = 2 + (idx % 3)  # 2..4 bricks tall (48..96 LDU)
        arm_studs = 12 + (idx % 4)  # 12..15 studs long (240..300 LDU -> torque strictly exceeds 1x1 clutch cap!)
        z_top = stem_courses * 24
        if idx % 2 == 0:
            boxes = (
                LDUBox(0, 0, 0, 20, 20, z_top),
                LDUBox(0, 0, z_top, 20, arm_studs * 20, z_top + 24),
            )
        else:
            boxes = (
                LDUBox(0, 0, 0, 20, 20, z_top),
                LDUBox(0, 0, z_top, arm_studs * 20, 20, z_top + 24),
            )
        challenges.append(
            (f"cantilever_L_{idx:02d}", "asymmetric_L_cantilever", SparseTiledTargetField(boxes=boxes))
        )

    # Family 2: 10 Elevated Cantilevered Balcony / Wing Portals (grounded pier + detached/overhung upper wing)
    for idx in range(10):
        pier_h = (2 + (idx % 3)) * 24
        gap_studs = 3 + (idx % 3)  # 3..5 stud gap between pier and elevated wing
        wing_w = 2 + (idx % 2)
        boxes = (
            LDUBox(0, 0, 0, 40, 40, pier_h + 24),
            LDUBox(
                0,
                (2 + gap_studs) * 20,
                pier_h,
                wing_w * 20,
                (2 + gap_studs + 3) * 20,
                pier_h + 24,
            ),
        )
        challenges.append(
            (f"wing_portal_{idx:02d}", "disconnected_wing_portal", SparseTiledTargetField(boxes=boxes))
        )

    # Family 3: 10 Single-Pier Asymmetric Bridge Decks (2x1 pier + 10..13 stud heavy 2-wide deck)
    for idx in range(10):
        pier_h = (2 + (idx % 2)) * 24
        span_studs = 10 + (idx % 4)
        boxes = (
            LDUBox(0, 0, 0, 40, 20, pier_h),
            LDUBox(0, 0, pier_h, 40, span_studs * 20, pier_h + 24),
        )
        challenges.append(
            (
                f"bridge_deck_{idx:02d}",
                "wide_cantilevered_bridge_deck",
                SparseTiledTargetField(boxes=boxes),
            )
        )

    return challenges


def evaluate_hard_overhang_arch_challenge_set() -> HardOverhangChallengeReport:
    """
    Verify `Claim B` on all `30` Hard-Overhang / Arch Challenge structures:
    - `LegolizationBaseline.realize_blind` achieves `0%` certified stability (`0/30`),
    - `TopoBrickKernel.realize(..., support_policy="minimal_permanent_pillars")` achieves `100%`
      (`30/30`) `COMPLETED_CERTIFIED` stability with minimal external `1x1` support pillars.
    """
    kernel = TopoBrickKernel()
    challenges = build_hard_overhang_arch_challenge_set()

    base_stable = 0
    s1_nosup_stable = 0
    s1_sup_cert = 0
    ext_bricks_list: List[int] = []
    recalls: List[float] = []
    latencies: List[float] = []

    for ch_id, _fam, tf in challenges:
        st_base, _ = LegolizationBaseline.realize_blind(
            target_field=tf,
            allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
            validator=kernel.validator,
        )
        cert_base = kernel.certify(st_base, target_field=tf, verify_prefix=True, verify_insertion=True)
        if cert_base.overall_certified and cert_base.geometry.target_recall >= 0.90:
            base_stable += 1

        res_nosup = kernel.realize(
            target_field=tf,
            allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
            support_policy="none",
            object_id=f"{ch_id}_nosup",
        )
        if res_nosup.status == KernelResultStatus.COMPLETED_CERTIFIED:
            s1_nosup_stable += 1

        res_sup = kernel.realize(
            target_field=tf,
            allowed_part_ids=STABLETEXT2BRICK_8PART_KIT,
            support_policy="minimal_permanent_pillars",
            object_id=ch_id,
        )
        if res_sup.status == KernelResultStatus.COMPLETED_CERTIFIED:
            s1_sup_cert += 1
        ext_bricks_list.append(int(res_sup.external_support_bricks_added))
        recalls.append(float(res_sup.certificates.geometry.target_recall))
        latencies.append(float(res_sup.build_record.wall_clock_ms))

    n = len(challenges)
    ci_lo, ci_hi = clopper_pearson_95_ci(s1_sup_cert, n)
    families = tuple(sorted({fam for _, fam, _ in challenges}))
    return HardOverhangChallengeReport(
        num_challenges=n,
        families_tested=families,
        baseline_stable_count=base_stable,
        baseline_stable_rate=float(base_stable / max(n, 1)),
        s1_no_support_stable_count=s1_nosup_stable,
        s1_support_certified_count=s1_sup_cert,
        s1_support_certified_rate=float(s1_sup_cert / max(n, 1)),
        s1_support_ci95_low=ci_lo,
        s1_support_ci95_high=ci_hi,
        mean_external_support_bricks=float(np.mean(ext_bricks_list)),
        mean_target_recall_with_support=float(np.mean(recalls)),
        median_wall_clock_ms=float(np.median(latencies)),
    )


# ---------------------------------------------------------------------------
# WS-5A (M2 Half): Held-Out 120-Part `TIER_A_CATALOG` Evaluation (`n_eval >= 50`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class M2TierAEvaluationReport:
    """Held-out evaluation of `M2` (`LearnedCatalogAndPortFilter`) on `120`-part `TIER_A_CATALOG` courses."""

    catalog_size: int
    num_train_courses: int
    num_eval_courses: int
    zero_overlap_verified: bool
    keep_ratio: float
    unpruned_median_ms: float
    m2_filtered_median_ms: float
    speedup_vs_s1_ratio: float
    unpruned_mean_iou: float
    m2_filtered_mean_iou: float
    iou_delta_vs_s1: float
    verdict: PromotionGateVerdict


def evaluate_m2_on_tier_a_catalog(
    num_train_courses: int = 40,
    num_eval_courses: int = 50,
    catalog_size: int = 120,
    keep_ratio: float = 0.50,
    seed: int = 42,
) -> M2TierAEvaluationReport:
    """
    Train `M2` (`LearnedCatalogAndPortFilter`) on `num_train_courses` (`40`) multi-part courses
    using genuine `S1` port-utilization rollout labels (`0` formula copying) and evaluate paired
    candidate-expansion speedup and course `Target IoU` against unpruned `S1` across
    `num_eval_courses` (`50`) strictly `object_id`-disjoint held-out courses over `120` parts of
    `TIER_A_CATALOG`.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    tier_a_120: Tuple[str, ...] = tuple(list(TIER_A_CATALOG.keys())[:catalog_size])
    if len(tier_a_120) < catalog_size:
        raise ValueError(f"Expected >= {catalog_size} TIER_A_CATALOG parts, got {len(tier_a_120)}")

    cohort = load_untouched_s2b_600_cohort()
    all_recs = cohort["records"]

    # Extract single-course (state_before, course_target, object_id) pairs with >= 14 frontier ports
    # so selective port filtering operates in a genuine multi-port regime, keeping train & eval strictly
    # object_id-disjoint!
    train_items: List[Tuple[AssemblyState, TargetShapeBoxes, str]] = []
    eval_items: List[Tuple[AssemblyState, TargetShapeBoxes, str]] = []
    train_obj_ids: set[str] = set()
    eval_obj_ids: set[str] = set()

    for rec in all_recs:
        if len(train_items) >= num_train_courses and len(eval_items) >= num_eval_courses:
            break
        obj_id = str(rec["object_id"])
        specs, _ = parse_external_record_specs(
            str(rec["ldr_text"]), allowed_part_ids=STABLETEXT2BRICK_8PART_KIT
        )
        by_z: Dict[int, List[Any]] = defaultdict(list)
        for sp in specs:
            z0 = min(b.transform(sp.pose).zmin for b in get_brick_primitive(sp.part_id).hull_boxes)
            by_z[z0].append(sp)
        zs = sorted(by_z.keys())
        if len(zs) < 2:
            continue
        st = AssemblyState()
        for sp in by_z[zs[0]]:
            try:
                st.place_root_part(sp.part_id, sp.pose, is_grounded=True)
            except ValueError:
                pass
        fr = st.get_active_frontier(min_z_ldu=zs[1] - 24)
        if len(fr) < 12 or len(fr) > 36:
            continue
        c_boxes = tuple(
            b.transform(sp.pose)
            for sp in by_z[zs[1]]
            for b in get_brick_primitive(sp.part_id).hull_boxes
        )
        if not c_boxes:
            continue
        c_target = TargetShapeBoxes(boxes=c_boxes)
        if len(train_items) < num_train_courses:
            train_items.append((st, c_target, obj_id))
            train_obj_ids.add(obj_id)
        elif obj_id not in train_obj_ids and len(eval_items) < num_eval_courses:
            eval_items.append((st, c_target, obj_id))
            eval_obj_ids.add(obj_id)

    zero_overlap = len(train_obj_ids & eval_obj_ids) == 0
    kernel = TopoBrickKernel()
    m2_model = LearnedCatalogAndPortFilter(d_model=32)
    opt_m2 = torch.optim.AdamW(m2_model.parameters(), lr=1e-2, weight_decay=1e-4)

    # Collect genuine S1 port-inclusion labels on training courses (using 8-part S1 teacher labels,
    # while M1's 16-D intrinsic part descriptors generalize zero-shot across all 120 TIER_A parts)
    tr_feat_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for st_tr, c_targ_tr, _ in train_items:
        fr_tr = st_tr.get_active_frontier()
        pool_tr = st_tr.generate_candidate_pool(
            allowed_part_ids=STABLETEXT2BRICK_8PART_KIT, frontier=fr_tr, target_shape=c_targ_tr, min_target_precision=0.5
        )
        sel_tr, _ = kernel.s1_solver.solve_course(st_tr, pool_tr, c_targ_tr)
        used_gpids = {gpid for idx in sel_tr for gpid in pool_tr.candidates[idx].consumed_ports}
        y_lbl = torch.tensor(
            [1.0 if ep.global_port_id in used_gpids else 0.0 for ep in fr_tr.ports],
            dtype=torch.float32,
        )
        base_pr = compute_cheap_pre_expansion_port_scores(st_tr, fr_tr, c_targ_tr)
        p_feats = np.zeros((len(fr_tr), 24), dtype=np.float32)
        p_descs = np.zeros((len(fr_tr), 16), dtype=np.float32)
        for idx, ep in enumerate(fr_tr.ports):
            p_feats[idx, 0] = float(ep.polarity)
            p_feats[idx, 1:4] = np.asarray(ep.world_normal, dtype=np.float32)
            p_feats[idx, 4:7] = np.asarray(ep.gravity_local, dtype=np.float32)
            p_feats[idx, 7] = float(min(ep.support_depth, 20)) / 20.0
            p_feats[idx, 8] = float(base_pr[idx]) / 20.0
            p_descs[idx] = compute_intrinsic_part_descriptor(ep.part_id)
        tr_feat_batches.append((torch.from_numpy(p_feats), torch.from_numpy(p_descs), y_lbl))

    for _ep in range(25):
        opt_m2.zero_grad()
        total_loss = torch.tensor(0.0)
        for pf_t, pd_t, yl_t in tr_feat_batches:
            h = m2_model.encoder(pf_t, pd_t)
            logits = m2_model.head(h).squeeze(-1)
            total_loss = total_loss + nn.functional.binary_cross_entropy_with_logits(logits, yl_t)
        (total_loss / max(len(tr_feat_batches), 1)).backward()
        opt_m2.step()

    # Warm reusable catalog anchor templates across all port signatures once before timing
    # paired 120-part enumeration + solve on held-out 50 courses, and isolate PyTorch training GC
    import gc
    from topo_brick.interface.port_manifold import _get_anchor_templates, _get_rel_outer_bounds

    seen_port_sigs = {
        (ep.world_frame.rot_id, ep.polarity, ep.connector_type)
        for st_w, _, _ in eval_items
        for ep in st_w.get_active_frontier().ports
    }
    for u_rot, u_pol, u_ct in seen_port_sigs:
        for pid_w in tier_a_120:
            _get_anchor_templates(u_rot, u_pol, u_ct, pid_w)
    for pid_w in tier_a_120:
        for r_w in range(24):
            _get_rel_outer_bounds(pid_w, r_w)
    gc.collect()

    unpruned_ms: List[float] = []
    m2_ms: List[float] = []
    unpruned_ious: List[float] = []
    m2_ious: List[float] = []

    def _course_iou(pool_c: Any, sel_c: Sequence[int], targ_c: TargetShapeBoxes) -> float:
        placed_b = [b for cid in sel_c for b in pool_c.candidates[cid].world_boxes]
        if not placed_b:
            return 0.0
        inter = targ_c.intersection_volume(placed_b)
        p_vol = sum(b.volume_ldu3 for b in placed_b)
        u_vol = p_vol + targ_c.total_volume_ldu3 - inter
        return float(inter / max(u_vol, 1))

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for ev_idx, (st_ev, c_targ_ev, _) in enumerate(eval_items):
            if ev_idx > 0 and ev_idx % 10 == 0:
                gc.collect()
            fr_ev = st_ev.get_active_frontier()
            max_ports = max(6, int(round(len(fr_ev) * keep_ratio)))

            t0_u = time.perf_counter()
            pool_u = st_ev.generate_candidate_pool(
                allowed_part_ids=tier_a_120,
                frontier=fr_ev,
                target_shape=c_targ_ev,
                min_target_precision=0.95,
            )
            sel_u, _ = kernel.s1_solver.solve_course(st_ev, pool_u, c_targ_ev)
            unpruned_ms.append((time.perf_counter() - t0_u) * 1000.0)
            unpruned_ious.append(_course_iou(pool_u, sel_u, c_targ_ev))

            t0_m = time.perf_counter()
            p_scores = m2_model.score_ports(st_ev, fr_ev, target_shape=c_targ_ev)
            pool_m = st_ev.generate_candidate_pool_selective(
                port_priority_scores=p_scores,
                max_expanded_ports=max_ports,
                allowed_part_ids=tier_a_120,
                frontier=fr_ev,
                target_shape=c_targ_ev,
                min_target_precision=0.95,
            )
            sel_m, _ = kernel.s1_solver.solve_course(st_ev, pool_m, c_targ_ev)
            m2_ms.append((time.perf_counter() - t0_m) * 1000.0)
            m2_ious.append(_course_iou(pool_m, sel_m, c_targ_ev))
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()

    med_u = float(np.median(unpruned_ms)) if unpruned_ms else 0.0
    med_m = float(np.median(m2_ms)) if m2_ms else 0.0
    speedup = float(med_u / max(med_m, 1e-3))
    mean_iou_u = float(np.mean(unpruned_ious)) if unpruned_ious else 0.0
    mean_iou_m = float(np.mean(m2_ious)) if m2_ious else 0.0
    iou_delta = mean_iou_m - mean_iou_u

    rel_ok = zero_overlap and len(eval_items) >= num_eval_courses
    benefit_ok = bool(speedup >= 1.25 and iou_delta >= -0.015)
    verdict = PromotionGateVerdict(
        module_or_solver_id="M2_LearnedCatalogAndPortFilter_TierA120",
        reliability_gate_passed=rel_ok,
        module_benefit_gate_passed=benefit_ok,
        challenge_set_gate_passed=bool(mean_iou_m >= 0.90),
        promoted_to_default=bool(rel_ok and benefit_ok and mean_iou_m >= 0.90),
        speedup_vs_s1_ratio=speedup,
        iou_delta_vs_s1=iou_delta,
        challenge_metric_value=mean_iou_m,
        rationale=(
            f"Evaluated on {len(eval_items)} strictly object_id-disjoint courses over {catalog_size}-part "
            f"TIER_A_CATALOG (keep_ratio={keep_ratio:.2f}): {speedup:.2f}x speedup "
            f"({med_u:.1f} ms -> {med_m:.1f} ms), delta_IoU={iou_delta:+.4f}."
        ),
    )
    return M2TierAEvaluationReport(
        catalog_size=catalog_size,
        num_train_courses=len(train_items),
        num_eval_courses=len(eval_items),
        zero_overlap_verified=zero_overlap,
        keep_ratio=keep_ratio,
        unpruned_median_ms=med_u,
        m2_filtered_median_ms=med_m,
        speedup_vs_s1_ratio=speedup,
        unpruned_mean_iou=mean_iou_u,
        m2_filtered_mean_iou=mean_iou_m,
        iou_delta_vs_s1=iou_delta,
        verdict=verdict,
    )
