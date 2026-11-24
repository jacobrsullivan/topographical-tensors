"""
Phase 12 (`v5.2.0`) Contract, Metamorphic & Falsification Audit Suite (`F1`–`F12`, `§5.2`, `§7.1`–`§7.4`):

Fast (`< 5 s`) deterministic regression tests locking down every finding and reproduction in the
21-Nov external code assessment so none of `F1`–`F12` can ever regress.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import List

import numpy as np
import pytest
import torch

from topo_brick import (
    BRICKGPT_VOCAB_SIZE,
    CATALOG_RETRIEVAL_INDEX,
    COMPILED_LDRAW_CATALOG,
    SET_10698_COLORED_INVENTORY,
    SET_10698_OFFICIAL_INVENTORY,
    SET_10698_REBRICKABLE_MANIFEST,
    AdaptiveSurfaceTarget,
    AssemblyState,
    BrickGrammarLogitsProcessor,
    CertStatus,
    CompiledLDrawCatalog,
    ContinuousAssemblyState,
    ContinuousSE3Pose,
    ExternalPhysicalObservation,
    KernelResultStatus,
    LDUBox,
    LDUFrame,
    SparseTiledTargetField,
    TopoBrickKernel,
    VerificationObligations,
    check_monotone_insertion_corridor,
    export_certificate_artifact,
    load_frozen_80_unseen_mesh_corpus,
    realize_open_catalog_sculpted,
    run_live_transformer_constrained_generation_benchmark,
    verify_certificate_independently,
)
from topo_brick.evaluation.independent_verifier import verify_certificate_json_file
import topo_brick.evaluation.phase8_round9_gates as p8_mod
from topo_brick.interface.ldraw_shadow_catalog_compiler import _greedy_maximal_cuboid_cover


def test_f1_delayed_counterweight_prefix_fail_and_trace_deletion_rejected() -> None:
    """Reproduce §7.1a & §7.1b: prefix FAIL and trace deletion MUST both fail independent verification."""
    kernel = TopoBrickKernel()
    st = AssemblyState()
    st.place_root_part("3003", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st.place_root_part("3009", LDUFrame(0, (0, -80, 24)), is_grounded=False)
    st.place_root_part("3009", LDUFrame(0, (20, 0, 24)), is_grounded=False)
    cert = kernel.certify(st, foundation_policy="table_friction_only")
    assert cert.static_mechanics.status == CertStatus.PASS
    assert cert.prefix_mechanics.status == CertStatus.FAIL
    assert not cert.overall_certified

    # §7.1a: Export unchanged (prefix_mechanics=FAIL) -> verifier MUST reject!
    art = export_certificate_artifact(
        st, cert, validator=kernel.validator, foundation_policy="table_friction_only"
    )
    rep_unchanged = verify_certificate_independently(art)
    assert rep_unchanged.verified is False
    assert rep_unchanged.prefix_mechanics_valid is False

    # Claim PASS with unstable trace -> verifier MUST reject!
    claims_pass = {**art.claimed_certificates, "prefix_mechanics": "PASS"}
    rep_lied_claim = verify_certificate_independently(replace(art, claimed_certificates=claims_pass))
    assert rep_lied_claim.verified is False

    # §7.1b: Claim PASS and delete trace -> verifier MUST reject!
    rep_no_trace = verify_certificate_independently(
        replace(art, claimed_certificates=claims_pass, execution_operations=())
    )
    assert rep_no_trace.verified is False
    assert rep_no_trace.prefix_mechanics_valid is False


def test_f1_trace_replay_tamper_unknown_op_and_port_index_bounds(tmp_path: Path) -> None:
    """Verify F1 tamper cases: truncated ops, fake op poses/parts, unknown op, empty/FAIL claims, port OOB, malformed JSON."""
    kernel = TopoBrickKernel()
    st = AssemblyState()
    st.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    cert = kernel.certify(st)
    art = export_certificate_artifact(st, cert, validator=kernel.validator)
    assert verify_certificate_independently(art).verified is True

    # 1. Deleting the final operation leaves terminal replay != declared placed_parts
    rep_trunc = verify_certificate_independently(replace(art, execution_operations=art.execution_operations[:1]))
    assert rep_trunc.verified is False
    assert any("trace_terminal_instance_mismatch" in n for n in rep_trunc.failure_reasons)

    # 2. Replacing operation poses with [999, 999, 999] and part_ids with "3005"
    fake_ops = tuple(
        {**op, "part_id": "3005", "pos": [999, 999, 999]} for op in art.execution_operations
    )
    rep_fake_ops = verify_certificate_independently(replace(art, execution_operations=fake_ops))
    assert rep_fake_ops.verified is False

    # 3. Unknown operation type
    rep_unk_op = verify_certificate_independently(
        replace(art, execution_operations=({"op_type": "TELEPORT", "instance_ids": [0]},))
    )
    assert rep_unk_op.verified is False
    assert any("unknown_op_type:TELEPORT" in n for n in rep_unk_op.failure_reasons)

    # 4. Removing all claims or replacing all claims with FAIL
    assert verify_certificate_independently(replace(art, claimed_certificates={})).verified is False
    all_fail = {k: "FAIL" for k in art.claimed_certificates}
    assert verify_certificate_independently(replace(art, claimed_certificates=all_fail)).verified is False

    # 5. Out-of-range matched port index returns structured failure (zero uncaught IndexError)
    bad_ports = (((0, 9999), (1, 0)),)
    rep_oob = verify_certificate_independently(replace(art, matched_port_pairs=bad_ports))
    assert rep_oob.verified is False
    assert any("invalid_port_index" in n for n in rep_oob.failure_reasons)

    # 6. Malformed JSON files return structured failure (zero uncaught exceptions)
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    assert verify_certificate_json_file(bad_json).verified is False

    list_json = tmp_path / "list.json"
    list_json.write_text("[1, 2, 3]", encoding="utf-8")
    assert verify_certificate_json_file(list_json).verified is False


def test_f2_continuous_floating_brick_table_policy_and_mode_validation() -> None:
    """Reproduce §7.2: floating continuous z=100 brick rejected under table_friction_only; unknown mode rejected."""
    kernel = TopoBrickKernel()
    cs_float = ContinuousAssemblyState()
    cs_float.place_part("3001", ContinuousSE3Pose.identity((0.0, 0.0, 100.0)), is_grounded=True)

    cert_table = kernel.certify(cs_float, foundation_policy="table_friction_only")
    assert cert_table.overall_certified is False
    assert cert_table.static_mechanics.status == CertStatus.FAIL

    art_table = export_certificate_artifact(
        cs_float, cert_table, foundation_policy="table_friction_only"
    )
    # Even if attacker lies in claims and deletes trace, floating brick on table_friction_only fails!
    lied_claims = {k: "PASS" for k in art_table.claimed_certificates}
    rep_float = verify_certificate_independently(
        replace(art_table, claimed_certificates=lied_claims, execution_operations=())
    )
    assert rep_float.verified is False

    # Under explicit FIXED_ANCHOR wall mount, elevated fixture is valid
    cert_anchor = kernel.certify(cs_float, foundation_policy="fixed_anchor")
    assert cert_anchor.overall_certified is True
    art_anchor = export_certificate_artifact(cs_float, cert_anchor, foundation_policy="fixed_anchor")
    assert verify_certificate_independently(art_anchor).verified is True

    # Unknown assembly_mode MUST be rejected
    rep_bad_mode = verify_certificate_independently(replace(art_anchor, assembly_mode="not_a_mode"))
    assert rep_bad_mode.verified is False
    assert any("unsupported_assembly_mode:not_a_mode" in n for n in rep_bad_mode.failure_reasons)


def test_f2_verifier_zero_transitive_kernel_or_static_equilibrium_imports(tmp_path: Path) -> None:
    """Verify in a fresh Python subprocess that continuous + cubic verification loads 0 kernel or static_equilibrium modules."""
    src_dir = str(Path(p8_mod.__file__).resolve().parents[2])
    art_file = tmp_path / "art_f2_continuous.json"
    sub_env = {**os.environ, "PYTHONPATH": src_dir}
    code = """
import json, pathlib, sys
from topo_brick import ContinuousAssemblyState, ContinuousSE3Pose, TopoBrickKernel, export_certificate_artifact
k = TopoBrickKernel()
cs = ContinuousAssemblyState()
b0 = cs.place_part("60471", ContinuousSE3Pose.identity((0.0, 0.0, 0.0)), is_grounded=True)
cs.attach_articulated_part(b0, 4, "92582", 6, joint_angle_rad=0.25)
cert = k.certify(cs, foundation_policy="clutched_baseplate", verify_perturbation=True)
art = export_certificate_artifact(cs, cert, foundation_policy="clutched_baseplate")
pathlib.Path(sys.argv[1]).write_text(json.dumps(art.to_json_dict()), encoding="utf-8")
"""
    gen_proc = subprocess.run(
        [sys.executable, "-c", code, str(art_file)],
        env=sub_env,
        capture_output=True,
        text=True,
    )
    assert gen_proc.returncode == 0, f"gen_proc failed:\nstderr={gen_proc.stderr}\nstdout={gen_proc.stdout}"

    verify_code = """
import json, pathlib, sys
from topo_brick.evaluation.independent_verifier import VersionedCertificateArtifact, verify_certificate_independently
raw = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
art = VersionedCertificateArtifact.from_json_dict(json.loads(raw))
rep = verify_certificate_independently(art)
forbidden = [
    m for m in sys.modules
    if m.startswith("topo_brick.kernel") or m == "topo_brick.evaluation.static_equilibrium"
]
print("JSON_RESULT:" + json.dumps({"verified": rep.verified, "forbidden": forbidden, "notes": list(rep.failure_reasons)}))
"""
    ver_proc = subprocess.run(
        [sys.executable, "-c", verify_code, str(art_file)],
        env=sub_env,
        capture_output=True,
        text=True,
    )
    assert ver_proc.returncode == 0, f"ver_proc failed:\nstderr={ver_proc.stderr}\nstdout={ver_proc.stdout}"
    json_lines = [ln[len("JSON_RESULT:") :] for ln in ver_proc.stdout.splitlines() if ln.startswith("JSON_RESULT:")]
    assert json_lines, f"Missing JSON_RESULT in stdout:\n{ver_proc.stdout}"
    out = json.loads(json_lines[-1])
    assert out["verified"] is True, out
    assert out["forbidden"] == [], f"Transitive forbidden modules loaded: {out['forbidden']}"


def test_f3_constraint_propagation_across_noop_repair_and_continuous_certify() -> None:
    """Reproduce §7.4: zero/insufficient inventory & forbidden boxes must be enforced on no-op repair & continuous certify."""
    kernel = TopoBrickKernel()
    st = AssemblyState()
    st.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)

    rep_zero = kernel.repair(st, inventory_counts={"3001": 0})
    assert rep_zero.status == KernelResultStatus.PROVEN_INFEASIBLE
    assert rep_zero.audit.inventory_respected is False
    assert rep_zero.repaired_certificates.inventory.status == CertStatus.FAIL

    # Forbidden support box intersecting already-stable brick must NOT pass no-op repair when locked
    rep_forb = kernel.repair(
        st,
        locked_instance_ids=frozenset({0}),
        forbidden_support_boxes=(LDUBox(-5, -5, 0, 5, 5, 24),),
    )
    assert rep_forb.status == KernelResultStatus.PROVEN_INFEASIBLE

    # Continuous certify with depleted inventory must fail InventoryCert & overall_certified
    cs = ContinuousAssemblyState()
    cs.place_part("3001", ContinuousSE3Pose.identity((0.0, 0.0, 0.0)), is_grounded=True)
    c_cont = kernel.certify(cs, inventory_counts={"3001": 0})
    assert c_cont.overall_certified is False
    assert c_cont.inventory.status == CertStatus.FAIL


def test_f4_frozen_80_mesh_corpus_strict_80_unique_geometries_and_live_funnel() -> None:
    """Reproduce §7.3: all 80 meshes in load_frozen_80_unseen_mesh_corpus have unique SHA-256 geometry hashes."""
    corpus, funnel = load_frozen_80_unseen_mesh_corpus()
    hashes = set()
    for m in corpus:
        h = hashlib.sha256()
        h.update(np.asarray(m["verts"], dtype=np.float64).tobytes())
        h.update(np.asarray(m["faces"], dtype=np.int32).tobytes())
        h.update(repr((m["max_xy_studs"], m["max_z_courses"], m["required_openings_ldu"])).encode())
        hashes.add(h.hexdigest())
    assert len(corpus) == 80
    assert len(hashes) == 80
    assert funnel["unique_geometry_sha256_count"] == 80
    assert funnel["excluded_non_manifold_or_empty"] == 2
    assert funnel["zero_prior_tuning_overlap_verified"] is True
    for band_cnt in funnel["band_unique_geometry_counts"].values():
        assert band_cnt == 20


def test_f5_rebrickable_set_10698_exact_joint_manifest_conservation() -> None:
    """Verify F5: Set 10698-1 colored inventory is loaded directly from the attributed Rebrickable manifest."""
    assert SET_10698_REBRICKABLE_MANIFEST["set_num"] == "10698-1"
    assert SET_10698_REBRICKABLE_MANIFEST["rebrickable_inventory_id"] == 102864
    assert SET_10698_REBRICKABLE_MANIFEST["summary"]["non_spare_total_pieces"] == 790
    assert SET_10698_COLORED_INVENTORY.total_pieces() == 790
    assert len(SET_10698_COLORED_INVENTORY.distinct_colors()) == 33
    assert SET_10698_COLORED_INVENTORY.marginal_mold_counts() == dict(SET_10698_OFFICIAL_INVENTORY)


def test_f6_gate5_matched_kit_no_proxy_tokens_and_random_transformer_scope() -> None:
    """Verify F6: zero dummy MLP scorer in phase8_round9_gates.py and explicit random-weights scope on BrickGPT logits report."""
    p8_src = Path(p8_mod.__file__).read_text(encoding="utf-8")
    assert "agent_scorer" not in p8_src
    tree = ast.parse(p8_src)
    fn_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "run_gate5_bricklayer_agent_and_m3_controls"
    ]
    assert len(fn_nodes) == 1

    sample_rec = {
        "object_id": "p12_smoke_00",
        "ldr_text": "1 15 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat\n",
    }
    rep_lm = run_live_transformer_constrained_generation_benchmark(records=[sample_rec], max_bricks=4)
    assert rep_lm.evaluation_scope == "bpe_logits_mask_integration_random_weights"


def test_f7_asset_tamper_detection_and_target_obligation_binding(tmp_path: Path) -> None:
    """Verify F7: target_sha256 obligation binding + direct byte SHA-256 verification on catalog & CoACD archives."""
    kernel = TopoBrickKernel()
    tf = SparseTiledTargetField(boxes=(LDUBox(0, 0, 0, 40, 80, 24),))
    st = AssemblyState()
    st.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    cert = kernel.certify(st, target_field=tf)
    art = export_certificate_artifact(st, cert, validator=kernel.validator, target_field=tf)

    # Without expected_target_field, verification_scope explicitly discloses assembly-only verification
    rep_asm_only = verify_certificate_independently(art)
    assert rep_asm_only.verified is True
    assert rep_asm_only.target_verified is None
    assert rep_asm_only.verification_scope == "assembly_only"

    # With expected_target_field obligation, target_verified is True, and corrupting target_sha256 fails!
    ob_tf = VerificationObligations(expected_target_field=tf)
    rep_bound = verify_certificate_independently(art, obligations=ob_tf)
    assert rep_bound.verified is True
    assert rep_bound.target_verified is True
    assert rep_bound.verification_scope == "assembly_and_target"

    rep_bad_target = verify_certificate_independently(
        replace(art, target_sha256="0" * 64), obligations=ob_tf
    )
    assert rep_bad_target.verified is False
    assert any("target_sha256_mismatch" in n for n in rep_bad_target.failure_reasons)

    # Tampered catalog archive with forged manifest digest MUST raise ValueError on load!
    forged_cat_gz = tmp_path / "forged_catalog.json.gz"
    forged_coacd = tmp_path / "forged_coacd.npz"
    np.savez_compressed(forged_coacd, dummy=np.zeros((1,), dtype=np.float32))
    with gzip.open(forged_cat_gz, "wt", encoding="utf-8") as gz:
        json.dump(
            {
                "manifest": {
                    "catalog_gz_sha256": "a" * 64,
                    "coacd_npz_sha256": "b" * 64,
                },
                "parts": [],
            },
            gz,
        )
    forged_manifest = tmp_path / "forged_manifest.json"
    forged_manifest.write_text(
        json.dumps({"catalog_gz_sha256": "a" * 64, "coacd_npz_sha256": "b" * 64}),
        encoding="utf-8",
    )
    bad_cat = CompiledLDrawCatalog(
        catalog_gz_path=forged_cat_gz,
        coacd_npz_path=forged_coacd,
        manifest_path=forged_manifest,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bad_cat._ensure_loaded()


def test_f8_cuboid_cover_zero_dropped_voxels_on_32_islands_and_full_catalog() -> None:
    """Verify F8: 32 disconnected voxel islands passed to max_boxes=28 retain 100% of solid cells; catalog drops 0."""
    occ = np.zeros((16, 16, 4), dtype=bool)
    island_centers: List[tuple[int, int, int]] = []
    for idx in range(32):
        ix = (idx % 8) * 2
        iy = (idx // 8) * 2
        iz = idx % 3
        occ[ix, iy, iz] = True
        island_centers.append((ix, iy, iz))
    boxes = _greedy_maximal_cuboid_cover(
        occ, origin_ldu=(0, 0, 0), voxel_ldu=(4, 4, 4), max_boxes=28
    )
    for ix, iy, iz in island_centers:
        cx, cy, cz = ix * 4 + 2, iy * 4 + 2, iz * 4 + 2
        assert any(
            b.xmin <= cx <= b.xmax and b.ymin <= cy <= b.ymax and b.zmin <= cz <= b.zmax
            for b in boxes
        ), f"Occupied voxel island {(ix, iy, iz)} was dropped!"

    COMPILED_LDRAW_CATALOG._ensure_loaded()
    assert COMPILED_LDRAW_CATALOG.manifest.get("dropped_solid_voxels_total", -1) == 0


def test_f9_calibrated_substud_routes_to_model_agreement_not_physical_validation() -> None:
    """Verify F9: verify_calibrated_substud_physics populates ModelAgreementCert while keeping PhysicalValidationCert UNKNOWN."""
    kernel = TopoBrickKernel()
    st = AssemblyState()
    st.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)

    cert_pred = kernel.certify(st, verify_calibrated_substud_physics=True)
    assert cert_pred.overall_certified is True
    assert cert_pred.model_agreement.status == CertStatus.PASS
    assert cert_pred.physical_validation.status == CertStatus.UNKNOWN

    obs = ExternalPhysicalObservation(
        rig_or_simulator_id="cmu_stable_lego_bench_v1",
        run_sha256="f" * 64,
        observed_stable=True,
        notes="stack_2x4_run_01",
    )
    cert_obs = kernel.certify(st, external_physical_observation=obs)
    assert cert_obs.physical_validation.status == CertStatus.PASS


def test_f10_continuous_half_ray_corridor_catches_high_overhang_and_thin_bar() -> None:
    """Verify F10: continuous half-ray clearance catches obstacles beyond 48 LDU (z=120) and thin mid-step bars."""
    st = AssemblyState()
    # Grounded target slot at (0, 0, 0) surrounded on +-X, +-Y, -Z AND capped at z=120 (> 48 LDU!)
    st.place_root_part("3005", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    st.place_root_part("3005", LDUFrame(0, (-20, 0, 0)), is_grounded=True)
    st.place_root_part("3005", LDUFrame(0, (0, 20, 0)), is_grounded=True)
    st.place_root_part("3005", LDUFrame(0, (0, -20, 0)), is_grounded=True)
    st.place_root_part("3024", LDUFrame(0, (0, 0, -8)), is_grounded=True)
    # High roof at z = 120 LDU (completely invisible to old 8..48 LDU discrete sampling!)
    st.place_root_part("3005", LDUFrame(0, (0, 0, 120)), is_grounded=True)

    assert check_monotone_insertion_corridor(st, "3005", LDUFrame(0, (0, 0, 0))) is False

    # Removing the z=120 roof opens the +Z half-ray to +infinity!
    st_open = AssemblyState()
    st_open.place_root_part("3005", LDUFrame(0, (20, 0, 0)), is_grounded=True)
    st_open.place_root_part("3005", LDUFrame(0, (-20, 0, 0)), is_grounded=True)
    st_open.place_root_part("3005", LDUFrame(0, (0, 20, 0)), is_grounded=True)
    st_open.place_root_part("3005", LDUFrame(0, (0, -20, 0)), is_grounded=True)
    st_open.place_root_part("3024", LDUFrame(0, (0, 0, -8)), is_grounded=True)
    assert check_monotone_insertion_corridor(st_open, "3005", LDUFrame(0, (0, 0, 0))) is True


def test_f11_unclamped_top_k_and_funnel_metadata() -> None:
    """Verify F11: retrieve_top_k respects exact top_k=1 and top_k=5 without clamping, and exposes funnel_metadata()."""
    occ = np.ones((8, 8, 4), dtype=bool)
    r1 = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ, top_k=1)
    r5 = CATALOG_RETRIEVAL_INDEX.retrieve_top_k(occ, top_k=5)
    assert len(r1.retrieved_part_ids) == 1
    assert len(r5.retrieved_part_ids) == 5
    assert len(r1.essential_backbone_part_ids) >= 8

    meta = CATALOG_RETRIEVAL_INDEX.funnel_metadata()
    assert meta["combined_catalog_ids"] == 16085
    assert meta["compiled_catalog_ids"] == 16072
    assert meta["active_buildable_molds"] >= 350
    assert meta["pose_signatures"] >= 1000


def test_f12_logits_processor_batch_guard_multistream_and_concurrent_certify() -> None:
    """Verify F12: batch_size>1 guard + multi-stream support, 4-thread HiGHS isolation, stage timings, and SE(3) translation invariance."""
    kernel = TopoBrickKernel()

    # 1. Single-state logits processor rejects batch_size > 1; multi-state handles batch_size == 2 cleanly
    proc_single = BrickGrammarLogitsProcessor(kernel, AssemblyState(), allowed_part_ids=("3001", "3003"))
    with pytest.raises(ValueError, match="requires batch_size == 1"):
        proc_single(torch.zeros((2, 1), dtype=torch.long), torch.zeros((2, 256), dtype=torch.float32))

    proc_batch = BrickGrammarLogitsProcessor(
        kernel, [AssemblyState(), AssemblyState()], allowed_part_ids=("3001", "3003")
    )
    masked = proc_batch(
        torch.zeros((2, 1), dtype=torch.long),
        torch.zeros((2, BRICKGPT_VOCAB_SIZE), dtype=torch.float32),
    )
    assert masked.shape == (2, BRICKGPT_VOCAB_SIZE)
    assert torch.isfinite(masked).any(dim=1).all()

    # 2. Stage-separated timings populated by modular open_catalog_stages pipeline
    occ2 = np.ones((8, 8, 12), dtype=bool)
    occ6 = np.ones((8, 8, 4), dtype=bool)
    ad_t = AdaptiveSurfaceTarget(
        nx_studs=2,
        ny_studs=2,
        nz_courses=1,
        occ_5x5x2=occ2,
        occ_5x5x6=occ6,
        mesh_surf_pts=np.zeros((0, 3), dtype=np.float64),
        rasterize_ms=0.1,
    )
    _, _, diag = realize_open_catalog_sculpted(
        kernel, SparseTiledTargetField(boxes=(LDUBox(0, 0, 0, 40, 40, 24),)), ad_t
    )
    assert set(diag.stage_timings_ms.keys()) == {
        "target_prep_ms",
        "retrieval_ms",
        "sculpt_synthesis_ms",
        "color_allocation_ms",
        "diagnostics_finalize_ms",
    }

    # 3. 4-thread concurrent certify() with thread-local HiGHS instances
    def _worker(idx: int) -> bool:
        st_w = AssemblyState()
        st_w.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
        if idx % 2 == 0:
            st_w.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)
            return kernel.certify(st_w, foundation_policy="table_friction_only").overall_certified is True
        st_w.place_root_part("3008", LDUFrame(0, (0, 80, 24)), is_grounded=False)
        return kernel.certify(st_w, foundation_policy="table_friction_only").overall_certified is False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_worker, range(24)))
    assert all(results)

    # 4. Metamorphic XY translation invariance vs +Z lift rejection on TABLE_FRICTION_ONLY
    st_a = AssemblyState()
    st_a.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_a.place_root_part("3003", LDUFrame(0, (0, 20, 24)), is_grounded=False)
    c_a = kernel.certify(st_a, foundation_policy="table_friction_only")

    st_b = AssemblyState()
    st_b.place_root_part("3001", LDUFrame(0, (200, -160, 0)), is_grounded=True)
    st_b.place_root_part("3003", LDUFrame(0, (200, -140, 24)), is_grounded=False)
    c_b = kernel.certify(st_b, foundation_policy="table_friction_only")
    assert c_a.overall_certified is True and c_b.overall_certified is True
    assert abs(c_a.static_mechanics.peak_utilization - c_b.static_mechanics.peak_utilization) < 1e-9

    st_z = AssemblyState()
    st_z.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=True)
    st_z.place_root_part("3003", LDUFrame(0, (0, 20, 48)), is_grounded=False)
    c_z = kernel.certify(st_z, foundation_policy="table_friction_only")
    assert c_z.overall_certified is False
