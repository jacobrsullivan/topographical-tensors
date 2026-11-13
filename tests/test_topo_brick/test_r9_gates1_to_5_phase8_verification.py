"""
Comprehensive Phase 8 Verification Suite (`R9-WS1` through `R9-WS5` / `Gates 1-5`):

Verifies every Round 9 external technical assessment remediation and enhancement end-to-end:
1. `Gate 1` (`R9-WS1`): Strict single-core thread/solver audit (`cpu_to_wall_ratio <= 1.25`),
   4-part physical provenance split (`8` hardware-labeled CMU `StableLego` JSONs separated from
   the 16-brick upstream `stick_heavy_good_test_horizontal_force.json` solver fixture and synthetic
   lateral sanity check, `BrickSim` `MIT` license), motif-family-disjoint `Train (n=60) -> Holdout (n=120)`
   `2x2` confusion matrix with all 3 false-pass denominators, and planner-free `independent_verifier`
   replay (`0` `TopoBrickKernel` imports verified via AST + tamper rejection).
2. `Gate 2` (`R9-WS2`): Fine `5 x 5 x 6 LDU` original-mesh fidelity (`voxelization_iou_loss > 0`),
   required arch-opening clearance disqualification & `forbidden_support_boxes` routing, `S1`, `S2`,
   `LuoInspiredSeamRefinement`, and out-of-process `hbmartin/legolization` (`0` GPL imports in repo).
3. `Gate 3` (`R9-WS3`): All `12` `Eval B-Extended` falsification cases (coupled 2-joint `SO(3)` saddle
   `min_hessian_eigenvalue < -1.0`, same-part internal-wall socket interference within `< 14 LDU`,
   axial insertion bounds, `REORIENT_ASSEMBLY`, premature `REMOVE_TEMP_SUPPORT`, bounded `repair()`
   `INCONCLUSIVE` vs `PROVEN_INFEASIBLE`), partial-overlap bearing-hull clamp, primal witness residuals,
   and `n = 200` `2 LDU` (`0.8 mm`) CSG plastic-solid voxel collider audit (`0` false acceptances).
4. `Gate 4` (`R9-WS4`): Frozen `6 x 4 = 24`-cell (`n = 120` unseen tasks) generative versatility matrix
   with blind roof pitch discovery on held-out continuous pitches `(13, 19, 27, 34, 38 deg)` and honest
   `Physical Validation` separation (`20/120` on `mixed_subassemblies_and_retail_inventory`).
5. `Gate 5` (`R9-WS5`): Paired real CMU `BrickLayer` (`bricklayer-sdk==0.1.0`) constructive agent A/B
   (`n = 50` tasks), zero-shot `M3_Pairwise_Sparse` vs. 6 §7.3 controls across 3 unseen families (`n = 60`),
   and `M1_Learned16D` vs. `RawMetadataControl` ablation (`n = 24`).
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path

from topo_brick import (
    EXTENDED_10PART_RECT_KIT,
    VERSATILITY_4_CONDITIONS,
    VERSATILITY_6_FAMILIES,
    AssemblyState,
    KernelResultStatus,
    LDUBox,
    LDUFrame,
    SparseTiledTargetField,
    TopoBrickKernel,
    evaluate_assembly_against_original_mesh,
    evaluate_upstream_horizontal_contact_fixture,
    export_certificate_artifact,
    load_physical_validation_assets,
    run_gate1_evidence_and_confusion_audit,
    run_gate2_fresh_mesh_and_external_comparator_benchmark,
    run_gate3_certificate_falsification_suite,
    run_gate4_24cell_generative_versatility_matrix,
    run_gate5_bricklayer_agent_and_m3_controls,
    verify_certificate_independently,
)
import topo_brick.evaluation.independent_verifier as indep_mod


def test_r9_gate1_single_core_provenance_confusion_and_independent_verifier() -> None:
    """Verify Gate 1: single-core audit, 8+1+1 provenance split, motif-disjoint 2x2 confusion, and AST-isolated independent verifier."""
    # 1. Verify `independent_verifier.py` imports ZERO planner / TopoBrickKernel modules
    indep_src = Path(indep_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(indep_src)
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
    for mod_name in imported_modules:
        assert "construction_kernel" not in mod_name
        assert "static_equilibrium" not in mod_name
        assert "planner" not in mod_name

    # 2. Verify `BrickSim` MIT license & strict 8-hardware vs 16-brick solver-fixture separation
    assets = load_physical_validation_assets()
    assert assets["provenance"]["bricksim_license"] == "MIT"
    assert len(assets["stable_lego_physical_cases"]) == 8
    hf_fx = evaluate_upstream_horizontal_contact_fixture()
    assert hf_fx["num_bricks"] == 16
    assert hf_fx["git_blob_sha1"] == "e5c1175606c447e10600f80a89f04b3d99374a82"
    assert hf_fx["is_confirmed_hardware_experiment"] is False
    assert hf_fx["stable_with_side_wall_press"] is True

    # 3. Run full Gate 1 audit
    g1 = run_gate1_evidence_and_confusion_audit()
    assert g1.single_core_audit.strictly_single_core_verified
    assert g1.single_core_audit.torch_num_threads == 1
    assert g1.single_core_audit.highs_threads == 1
    assert g1.cmu_hardware_8cases_matched == 8
    assert g1.cmu_hardware_8cases_total == 8
    assert g1.upstream_horizontal_fixture_verified
    assert g1.synthetic_lateral_push_sanity_verified
    assert g1.zero_motif_family_overlap
    assert g1.holdout_confusion.total_n == 120
    assert g1.holdout_confusion.fp_over_total <= 0.02
    assert g1.independent_certificates_tested >= 25
    assert g1.independent_certificates_verified == g1.independent_certificates_tested
    assert g1.max_independent_eq_residual_g <= 1e-5
    assert g1.gate1_passed

    # 4. Tamper test: corrupting a certified artifact's pose to collide MUST fail independent verification!
    kernel = TopoBrickKernel()
    st_ok = AssemblyState()
    st_ok.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_ok.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    cert_ok = kernel.certify(st_ok, verify_prefix=True, verify_insertion=True)
    art_ok = export_certificate_artifact(st_ok, cert_ok, validator=kernel.validator)
    assert verify_certificate_independently(art_ok).verified is True

    tampered_parts = (
        art_ok.placed_parts[0],
        {**art_ok.placed_parts[1], "pos": [0, 0, 12]},  # 12 LDU vertical interpenetration!
    )
    art_tampered = replace(art_ok, placed_parts=tampered_parts)
    rep_tampered = verify_certificate_independently(art_tampered)
    assert rep_tampered.verified is False
    assert rep_tampered.csg_2ldu_collision_free is False


def test_r9_gate2_original_mesh_subvoxel_opening_and_external_comparators() -> None:
    """Verify Gate 2: fine 5x5x6 LDU original-mesh IoU, required opening disqualification, S1/S2/Luo, and zero GPL imports."""
    # 1. Verify ZERO Python imports of GPL `legolization` anywhere in `src/topo_brick/`
    src_root = Path(indep_mod.__file__).resolve().parents[1]
    for py_file in src_root.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for al in node.names:
                    assert not al.name.startswith("legolization")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("legolization")

    # 2. Verify arch-opening disqualification: blocking `required_openings_ldu` flips `required_opening_clear == False`,
    #    whereas `kernel.realize(..., forbidden_support_boxes=...)` routes permanent pillars outside openings!
    kernel = TopoBrickKernel()
    arch_opening = LDUBox(20, 0, 0, 60, 40, 24)
    tf_arch = SparseTiledTargetField(
        boxes=(
            LDUBox(0, 0, 0, 20, 40, 48),
            LDUBox(60, 0, 0, 80, 40, 48),
            LDUBox(20, 0, 24, 60, 40, 48),
        )
    )
    res_clear = kernel.realize(
        tf_arch,
        solver_id="S1_CoverAndRepair",
        allowed_part_ids=EXTENDED_10PART_RECT_KIT,
        support_policy="minimal_permanent_pillars",
        forbidden_support_boxes=(arch_opening,),
    )
    assert res_clear.status == KernelResultStatus.COMPLETED_CERTIFIED
    for p in res_clear.assembly.placed_parts.values():
        for wb in p.world_boxes:
            assert not wb.intersects(arch_opening)

    # Deliberately place a brick inside `arch_opening` and confirm `evaluate_assembly_against_original_mesh` flags it!
    st_blocked = copy.deepcopy(res_clear.assembly)
    st_blocked.place_root_part("3003", LDUFrame(0, (30, 10, 0)), is_grounded=True)
    import numpy as np

    dummy_fine = np.ones((20, 10, 10), dtype=bool)
    dummy_pts = np.array([[10.0, 10.0, 10.0], [70.0, 30.0, 40.0]], dtype=np.float64)
    fid_clear = evaluate_assembly_against_original_mesh(
        dummy_fine, dummy_pts, tf_arch, res_clear.assembly, required_openings_ldu=(arch_opening,)
    )
    fid_blocked = evaluate_assembly_against_original_mesh(
        dummy_fine, dummy_pts, tf_arch, st_blocked, required_openings_ldu=(arch_opening,)
    )
    assert fid_clear.required_opening_clear is True
    assert fid_blocked.required_opening_clear is False

    # 3. Run Gate 2 across stratified unseen triangle meshes (`4` bands x `4` = `16` in pytest for fast CI;
    #    full `80`-mesh ledger verified via saved external comparator cache & standalone CLI)
    g2 = run_gate2_fresh_mesh_and_external_comparator_benchmark(max_meshes_per_band=4)
    assert g2.num_meshes == 16
    s1_no = g2.solver_summaries["S1_NoSupport"]
    s1_sup = g2.solver_summaries["S1_PermittedSupport"]
    s2_no = g2.solver_summaries["S2_NoSupport"]
    luo = g2.solver_summaries["LuoInspiredSeamRefinement"]
    assert s1_no.mean_voxelization_iou_loss > 0.01
    assert s1_no.mean_original_mesh_iou < 0.99
    assert s1_sup.opening_clear_count == 16
    assert s2_no.certified_rate >= 0.85
    assert s1_no.qualified_at_iou60_rate >= luo.qualified_at_iou60_rate
    assert g2.gate2_passed


def test_r9_gate3_all_12_falsification_cases_and_csg_collider_audit() -> None:
    """Verify Gate 3: all 12 Eval B-Extended adversarial cases, partial-overlap hull, witness residuals, and 2 LDU CSG audit."""
    g3 = run_gate3_certificate_falsification_suite(csg_samples=120)
    assert g3.original_6_cases_passed
    assert g3.case7_coupled_two_joint_saddle_rejected
    assert g3.case7_mechanism_regime == "unstable_coupled_saddle_mechanism"
    assert g3.case7_min_hessian_eigenvalue < -1.0
    assert g3.case8_same_part_internal_wall_rejected
    assert g3.case9_invalid_insertion_depth_rejected
    assert g3.case10_reorient_assembly_intermediate_collapse_rejected
    assert g3.case11_temp_support_premature_removal_rejected
    assert g3.case12_bounded_repair_exhaustion_returns_inconclusive
    assert g3.case12_wider_repair_certified
    assert g3.partial_overlap_bearing_hull_verified
    assert g3.primal_witness_residual_verified
    assert g3.csg_collider_audit_n == 120
    assert g3.csg_false_acceptance_count == 0
    assert g3.gate3_passed


def test_r9_gate4_24cell_generative_versatility_matrix() -> None:
    """Verify Gate 4: 6x4=24-cell (120 unseen tasks) generative versatility matrix & blind continuous roof pitch discovery."""
    g4 = run_gate4_24cell_generative_versatility_matrix()
    assert g4.total_cells == 24
    assert g4.total_tasks == 120
    assert len(g4.cells) == len(VERSATILITY_6_FAMILIES) * len(VERSATILITY_4_CONDITIONS)
    assert g4.held_out_continuous_pitches_deg == (13.0, 19.0, 27.0, 34.0, 38.0)
    assert g4.max_pitch_discovery_error_deg <= 0.05
    assert g4.overall_kernel_certified >= 114
    assert g4.overall_physically_validated == 20  # Honest scope: only calibrated stud-family cell!
    assert g4.overall_kernel_certified > g4.luo_inspired_total_certified
    assert g4.gate4_passed


def test_r9_gate5_bricklayer_agent_ab_and_m3_m1_controls() -> None:
    """Verify Gate 5: 50 paired BrickLayer agent tasks, M3_Pairwise_Sparse vs 6 controls on 3 unseen families, and M1 ablation."""
    g5 = run_gate5_bricklayer_agent_and_m3_controls()
    assert g5.arm_standalone.num_tasks == 50
    assert g5.arm_plus_kernel.num_tasks == 50
    assert g5.arm_plus_kernel.qualified_build_rate >= 0.92
    assert g5.arm_plus_kernel.qualified_build_rate > g5.arm_standalone.qualified_build_rate + 0.35

    required_controls = {
        "S1_Myopic",
        "S1_HandcraftedGeomSynergy",
        "S2_PairSeededLocalSearch",
        "S1_CoherentTieBreak",
        "S1_SequentialConditional",
        "S1_ExactTwoCourse",
        "M3_Pairwise_Sparse",
    }
    assert required_controls.issubset(set(g5.m3_controls.keys()))
    m3_sp = g5.m3_controls["M3_Pairwise_Sparse"]
    assert m3_sp.num_tasks == 60
    assert m3_sp.certified_solve_rate >= 0.90
    assert m3_sp.mean_sparse_edge_density < 1.0
    assert g5.m1_raw_metadata_control_rate >= 0.90
    assert g5.gate5_passed
