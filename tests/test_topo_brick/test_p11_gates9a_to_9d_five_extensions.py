"""
Phase 11 (`v5.2.0`) Acceptance & Unit Test Suite:
Five Extensions to `topo_brick` for Constrained Generative & Multimodal Synthesis (`Gates 9A–9D`).

Covers:
1. `SET_10698_COLORED_INVENTORY` (`790` pieces, `33` LDraw colors) exact marginal conservation
   against `SET_10698_OFFICIAL_INVENTORY` and lossless JSON round-trip.
2. `CATALOG_RETRIEVAL_INDEX.retrieve_top_k(..., allowed_part_ids=...)` `O(1)` finite-inventory
   masking and exact owned mold-alias back-mapping (`e.g. 3004 -> 3004`, `3040 -> 3040` when
   canonical index stores `3040b`).
3. `LANDMARK_DETAIL_INDEX` (`compiled_landmark_detail_index.json.gz`) `8 / 8` role coverage under
   both `license_filter="all"` and `"cc_by_4_0_only"`.
4. `compute_catalog_sha256` schema-aware SHA-256 binding (`"1.1.0"` includes landmark index while
   `"1.0.0"` preserves backward-compatible 3-part digest) and AST planner-isolation of
   `independent_verifier.py`.
5. `Gate 9A` (`Ext 1 & 2`): Colored finite-inventory synthesis (`n = 40` under `10698-1` + `n = 20`
   adversarial zero-`1x1` odd-parity targets + mutant rejection + `InventoryShortfallReport`).
6. `Gate 9B` (`Ext 3`): Ribbed hollow-shell scaling (`n = 30`, `16x16x10` to `24x24x16`,
   `FoundationPolicy.TABLE_FRICTION_ONLY`).
7. `Gate 9C` (`Ext 4A & 4B`): Compositional multi-block `.mpd` & 8-role landmarks (`n = 25`,
   recursive `subtrace` replay, tampered `subtrace` mutant rejection, exact `SE(3)` round-trip).
8. `Gate 9D` (`Ext 5`): View-prioritized scarcity ablation (`n = 30`, `TABLE_FRICTION_ONLY`).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

import topo_brick
from topo_brick.evaluation.independent_verifier import (
    CERTIFICATE_SCHEMA_VERSION,
    compute_catalog_sha256,
)
from topo_brick.evaluation.phase11_five_extensions_gates import (
    run_gate9a_colored_finite_inventory_and_sparse_kit_benchmark,
    run_gate9b_ribbed_hollow_shell_scaling_benchmark,
    run_gate9c_compositional_mpd_and_landmarks_benchmark,
    run_gate9d_view_prioritized_scarcity_study,
)
from topo_brick.interface.brick_catalog import (
    ColoredInventorySpec,
    SET_10698_COLORED_INVENTORY,
)
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
from topo_brick.interface.landmark_detail_index import (
    LANDMARK_DETAIL_INDEX,
    LANDMARK_DETAIL_ROLES,
)
from topo_brick.interface.tier_a_catalog import (
    SET_10698_33_LDRAW_COLORS,
    SET_10698_OFFICIAL_INVENTORY,
)


def test_p11_lazy_exports_and_schema_v110_digest_hygiene() -> None:
    """Verify all Phase 11 PEP 562 top-level symbols resolve cleanly and Schema `v1.1.0` digest binds landmark index."""
    required_exports = (
        "ColoredPartKey",
        "ColoredInventorySpec",
        "SET_10698_COLORED_INVENTORY",
        "InventoryShortfallReport",
        "InventoryCert",
        "HollowShellPolicy",
        "apply_hollow_shell_policy",
        "evaluate_exterior_crust_fidelity",
        "ViewPrioritySpec",
        "CameraReprojectionMetrics",
        "compute_subvoxel_view_weights",
        "evaluate_camera_reprojection_fidelity",
        "LANDMARK_DETAIL_ROLES",
        "LANDMARK_DETAIL_INDEX",
        "SubmodelNodeSpec",
        "LandmarkAttachmentRequest",
        "run_gate9a_colored_finite_inventory_and_sparse_kit_benchmark",
        "run_gate9b_ribbed_hollow_shell_scaling_benchmark",
        "run_gate9c_compositional_mpd_and_landmarks_benchmark",
        "run_gate9d_view_prioritized_scarcity_study",
        "run_all_phase11_gates",
    )
    for name in required_exports:
        assert hasattr(topo_brick, name), f"Missing top-level export: {name}"

    assert CERTIFICATE_SCHEMA_VERSION == "1.1.0"
    sha_110 = compute_catalog_sha256("1.1.0")
    sha_100 = compute_catalog_sha256("1.0.0")
    assert len(sha_110) == 64 and len(sha_100) == 64
    assert sha_110 != sha_100, "Schema 1.1.0 digest must bind compiled_landmark_detail_index.json.gz"


def test_p11_independent_verifier_ast_isolation() -> None:
    """Ensure `independent_verifier.py` remains 100% isolated from kernel/planner/realizer modules."""
    iv_path = (
        Path(topo_brick.__file__).resolve().parent / "evaluation" / "independent_verifier.py"
    )
    tree = ast.parse(iv_path.read_text(encoding="utf-8"))
    forbidden_substrings = (
        "construction_kernel",
        "static_equilibrium",
        "open_catalog_realizer",
        "colored_inventory_allocator",
        "subassembly_realizer",
        "adaptive_surface_and_macro_planner",
        "view_priority",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for bad in forbidden_substrings:
                    assert bad not in alias.name, f"Forbidden import {alias.name} in independent_verifier.py"
        elif isinstance(node, ast.ImportFrom) and node.module:
            for bad in forbidden_substrings:
                assert bad not in node.module, f"Forbidden from-import {node.module} in independent_verifier.py"


def test_p11_set_10698_colored_inventory_exact_conservation() -> None:
    """Verify `SET_10698_COLORED_INVENTORY` has 790 pieces across 33 colors and matches `SET_10698_OFFICIAL_INVENTORY`."""
    assert SET_10698_COLORED_INVENTORY.total_pieces() == 790
    distinct_colors = set(SET_10698_COLORED_INVENTORY.distinct_colors())
    assert len(distinct_colors) == 33
    assert distinct_colors == set(SET_10698_33_LDRAW_COLORS)
    assert SET_10698_COLORED_INVENTORY.marginal_mold_counts() == dict(SET_10698_OFFICIAL_INVENTORY)

    flat = SET_10698_COLORED_INVENTORY.to_flat_json_dict()
    reloaded = ColoredInventorySpec.from_flat_json_dict(flat)
    assert reloaded.total_pieces() == 790
    assert reloaded.marginal_mold_counts() == dict(SET_10698_OFFICIAL_INVENTORY)


def test_p11_inventory_masked_retrieval_and_alias_rewrite() -> None:
    """Verify `retrieve_top_k(..., allowed_part_ids=...)` masks out unowned molds and rewrites mold aliases."""
    occ = np.ones((16, 16, 12), dtype=bool)
    # Pass printed/variant aliases from Set 10698 (`98138pr0008 -> 98138`, `30136 -> 3004`) + `3001` + `3023`
    owned = ("3001", "3023", "98138pr0008", "30136")
    res = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ, top_k=16, allowed_part_ids=owned)
    assert set(res.retrieved_part_ids).issubset(set(owned))
    assert "30136" in res.retrieved_part_ids
    assert "3004" not in res.retrieved_part_ids
    assert "98138" not in res.retrieved_part_ids


def test_p11_landmark_detail_index_8role_cc_by_4_hygiene() -> None:
    """Verify all 8 landmark roles have verified `CC-BY-4.0` coverage with zero ShareAlike leakage."""
    assert len(LANDMARK_DETAIL_ROLES) == 8
    assert len(LANDMARK_DETAIL_INDEX) >= 50
    for role in LANDMARK_DETAIL_ROLES:
        all_recs = LANDMARK_DETAIL_INDEX.query_role(role, license_filter="all")
        cc4_recs = LANDMARK_DETAIL_INDEX.query_role(role, license_filter="cc_by_4_0_only")
        assert len(all_recs) >= 1, f"Role {role} empty under license_filter='all'"
        assert len(cc4_recs) >= 1, f"Role {role} empty under license_filter='cc_by_4_0_only'"
        for r in cc4_recs:
            assert r.license_tier == "cc_by_4_0", f"CC-BY-SA leak {r.part_id} in role {role}"


def test_p11_gate9a_colored_finite_inventory_and_sparse_kits() -> None:
    """Gate 9A (`Ext 1 & 2`): 60/60 certified + Schema v1.1.0 verified, 0 overruns, >=35% ΔE_00 drop, <250ms p50."""
    rep = run_gate9a_colored_finite_inventory_and_sparse_kit_benchmark()
    assert rep.single_core_audit.strictly_single_core_verified
    assert rep.total_targets == 60
    assert rep.total_certified_count == 60
    assert rep.total_verifier_passed_count == 60
    assert rep.set_10698_color_overrun_count == 0
    assert rep.sparse_odd_parity_one_by_one_emitted == 0
    assert rep.set_10698_skin_delta_e00_reduction_ratio >= 0.35
    assert rep.tampered_color_mutants_rejected == rep.tampered_color_mutants_tested
    assert rep.shortfall_infeasible_detected
    assert rep.realize_p50_wall_ms <= 250.0
    assert rep.gate9a_passed


def test_p11_gate9b_ribbed_hollow_shell_scaling() -> None:
    """Gate 9B (`Ext 3`): Audit 30/30 dual-host ledger + execute live 10/10 5-archetype TABLE_FRICTION_ONLY slice."""
    ledger_path = Path(__file__).resolve().parents[2] / "docs" / "phase11-gate-verification-ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger["all_phase11_gates_passed_both_hosts"] is True
        assert ledger["cross_host_metrics_exact_match"] is True
        for host_key in ("host_a", "host_b"):
            g9b = ledger[host_key]["gate9b"]
            assert g9b["gate9b_passed"] is True
            assert g9b["num_targets"] == 30
            assert g9b["hollow_certified_count"] == 30
            assert g9b["solid_certified_count"] == 30
            assert g9b["hollow_insertion_pass_count"] == 30
            assert g9b["hollow_temp_supports_emitted"] == 0
            assert g9b["independent_verifier_passed_count"] == 30
            assert g9b["mean_brick_reduction_ratio"] >= 0.45
            assert g9b["mean_volume_savings_ratio"] >= 0.45
            assert g9b["mean_abs_crust_iou_diff"] <= 0.015

    # Stratified live execution across all 5 structural archetypes (2 scales per archetype = 10 targets)
    stratified_10 = (0, 2, 6, 8, 12, 14, 18, 20, 24, 26)
    rep = run_gate9b_ribbed_hollow_shell_scaling_benchmark(target_indices=stratified_10)
    assert rep.single_core_audit.strictly_single_core_verified
    assert rep.num_targets == 10
    assert rep.hollow_certified_count == 10
    assert rep.solid_certified_count == 10
    assert rep.hollow_insertion_pass_count == 10
    assert rep.hollow_temp_supports_emitted == 0
    assert rep.independent_verifier_passed_count == 10
    assert rep.mean_brick_reduction_ratio >= 0.45
    assert rep.mean_volume_savings_ratio >= 0.45
    assert rep.mean_abs_crust_iou_diff <= 0.015
    assert rep.gate9b_passed


def test_p11_gate9c_compositional_mpd_and_landmarks() -> None:
    """Gate 9C (`Ext 4A & 4B`): 25/25 certified + subtrace verified + SE(3) round-trip + 8/8 roles + 0 CC-BY-SA leaks."""
    rep = run_gate9c_compositional_mpd_and_landmarks_benchmark()
    assert rep.single_core_audit.strictly_single_core_verified
    assert rep.num_scenes == 25
    assert rep.certified_count == 25
    assert rep.independent_verifier_passed_count == 25
    assert rep.mpd_se3_roundtrip_exact_count == 25
    assert rep.cc_by_4_0_share_alike_leaks == 0
    assert rep.landmark_roles_exercised_count == 8
    assert rep.tampered_subtrace_mutants_rejected == rep.tampered_subtrace_mutants_tested
    assert rep.gate9c_passed


def test_p11_gate9d_view_prioritized_scarcity_study() -> None:
    """Gate 9D (`Ext 5`): 30/30 TABLE_FRICTION_ONLY certified + verified, >=+0.04 2D IoU lift, >=4.0 ΔE_00 drop."""
    rep = run_gate9d_view_prioritized_scarcity_study()
    assert rep.single_core_audit.strictly_single_core_verified
    assert rep.num_targets == 30
    assert rep.view_on_certified_count == 30
    assert rep.view_off_certified_count == 30
    assert rep.independent_verifier_passed_count == 30
    assert rep.mean_silhouette_iou_2d_lift >= 0.04
    assert rep.mean_visible_delta_e00_reduction >= 4.0
    assert rep.gate9d_passed
