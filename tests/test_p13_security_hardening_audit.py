"""
Phase 13 (`v5.2.1`) Security Hardening Audit Suite (`S1`–`S10`).

Verifies all 10 findings from the security assessment end-to-end:
  - `S1` & `S2`: Authenticated `weights_only=True` checkpoint I/O, `python -O` digest enforcement
    prior to `torch.load`, pre-allocation schema/dimension caps, `0o700` run dirs, and `torch>=2.10.0`.
  - `S3`: `VerificationObligations.min_target_iou` never silently skipped (`cubic_ldu`, `continuous_se3`,
    digest-only, and bare `min_target_iou`), with separate `target_digest_verified` and
    `target_overlap_verified` fields.
  - `S4`: Relying-party `expected_inventory_counts`, `require_explicit_inventory_spec`,
    `expected_foundation_policy`, `MechanicsVerificationProfile` parameter/tolerance ceilings,
    `inventory_status`, and `mechanics_profile_sha256`.
  - `S5`: Bounded `VerifierResourceLimits`, `verify_certificate_in_isolated_worker` timeout containment,
    LDCad grid & DAG `_ExpansionBudget` caps, and Lichess CSV line-byte bounds.
  - `S6`: `StockfishDistiller` premature-EOF, hang-timeout, spam line-cap failure handling, child reaping,
    and child environment sanitization.
  - `S7`–`S10`: Separate `checkpoint_audit` provenance counts + `--expected-manifest`, fail-closed
    `CompiledLDrawCatalog` + pinned release catalog digest check, SHA-pinned CI workflows & lockfiles,
    and digest-pinned non-root `Dockerfile`.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import pickle
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import yaml

from topo_brick.evaluation.independent_verifier import (
    CERTIFICATE_SCHEMA_VERSION,
    MechanicsVerificationProfile,
    VerificationObligations,
    VerifierResourceLimits,
    VersionedCertificateArtifact,
    compute_catalog_sha256,
    compute_target_sha256,
    main as verifier_cli_main,
    verify_certificate_in_isolated_worker,
    verify_certificate_independently,
    verify_certificate_json_file,
)
from topo_brick.interface.ldraw_shadow_catalog_compiler import (
    CatalogResourceLimitError,
    CompiledLDrawCatalog,
    _ExpansionBudget,
    _expand_ldcad_grid,
)
from topo_brick.interface.brick_catalog import get_brick_primitive
from topo_brick.interface.ldu_frames import LDUBox, LDUFrame, rot_id_to_matrix
from topo_brick.interface.release_provenance import (
    CatalogProvenanceError,
    PINNED_RELEASE_ASSET_DIGESTS_V5_2,
    verify_bundled_asset_digests,
)
from topo_brick.representation.interface_fields import TargetShapeBoxes
from topo_chess.data.lichess_phase_c import (
    _iter_bounded_csv_chunks,
    _read_bounded_csv_line,
    build_or_load_phase_c_lichess_datasets,
)
from topo_chess.data.pipeline import StockfishDistiller, StockfishProtocolError
from topo_chess.domains.boards import Standard8x8State
from topo_chess.evaluation.phase_c_trainer_and_gauntlet import reload_phase_c_checkpoint_agent
from topo_chess.models.ablation_configs import AblationConfigAgent
from topo_chess.models.ipa_encoder import TopographicalIPAEncoder
from topo_chess.security.checkpoint_io import (
    CheckpointIntegrityError,
    CheckpointSchemaError,
    CheckpointSecurityError,
    UnsupportedTorchSecurityVersionError,
    create_private_checkpoint_dir,
    safe_load_authenticated_checkpoint,
    safe_save_checkpoint,
    validate_phase_c_checkpoint_schema,
    verify_safe_torch_version,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_valid_two_brick_cubic_artifact(
    *,
    target_sha256: str = "no_target",
    inventory_spec: dict[str, int] | None = None,
    foundation_policy: str = "clutched_baseplate",
) -> VersionedCertificateArtifact:
    return VersionedCertificateArtifact(
        schema_version=CERTIFICATE_SCHEMA_VERSION,
        catalog_sha256=compute_catalog_sha256(),
        target_sha256=target_sha256,
        assembly_mode="cubic_ldu",
        foundation_policy=foundation_policy,
        support_policy="none",
        inventory_spec=inventory_spec,
        placed_parts=(
            {
                "instance_id": 1,
                "part_id": "3001",
                "rot_id": 0,
                "pos": [0, 0, 0],
                "is_grounded": True,
                "color_id": 15,
            },
            {
                "instance_id": 2,
                "part_id": "3001",
                "rot_id": 0,
                "pos": [0, 0, 24],
                "is_grounded": False,
                "color_id": 15,
            },
        ),
        execution_operations=(
            {
                "op_type": "PLACE_PART",
                "instance_ids": [1],
                "part_id": "3001",
                "rot_id": 0,
                "pos": [0, 0, 0],
                "is_grounded": True,
                "color_id": 15,
                "temp_support_tag": "",
                "gravity_unit": [0.0, 0.0, -1.0],
                "subtrace": [],
            },
            {
                "op_type": "PLACE_PART",
                "instance_ids": [2],
                "part_id": "3001",
                "rot_id": 0,
                "pos": [0, 0, 24],
                "is_grounded": False,
                "color_id": 15,
                "temp_support_tag": "",
                "gravity_unit": [0.0, 0.0, -1.0],
                "subtrace": [],
            },
        ),
        matched_port_pairs=(((1, 0), (2, 8)),),
        joint_assumptions={
            "clutch_force_capacity_grams": 99.9,
            "moment_arm_capacity_ldu": 40.0,
            "friction_coeff": 0.40,
            "normal_crush_ratio": 500.0,
        },
        tolerances={
            "eq_residual_tol_g": 1e-4,
            "ub_violation_tol": 1e-4,
            "fine_csg_voxel_ldu": 2.0,
        },
        lp_witness_summary={"peak_joint_utilization": 0.03},
        cached_dependencies=(),
        claimed_certificates={
            "geometry": "PASS",
            "static_mechanics": "PASS",
            "prefix_mechanics": "PASS",
            "insertion": "PASS",
        },
    )


def _make_valid_single_brick_se3_artifact(target_sha256: str) -> VersionedCertificateArtifact:
    eye = rot_id_to_matrix(0).astype(float).tolist()
    return VersionedCertificateArtifact(
        schema_version=CERTIFICATE_SCHEMA_VERSION,
        catalog_sha256=compute_catalog_sha256(),
        target_sha256=target_sha256,
        assembly_mode="continuous_se3",
        foundation_policy="clutched_baseplate",
        support_policy="none",
        inventory_spec={"3001": 1},
        placed_parts=(
            {
                "instance_id": 1,
                "part_id": "3001",
                "rot_matrix": eye,
                "pos": [0.0, 0.0, 0.0],
                "is_grounded": True,
                "color_id": 15,
            },
        ),
        execution_operations=(
            {
                "op_type": "SE3_PLACE_PART",
                "instance_ids": [1],
                "part_id": "3001",
                "rot_id": 0,
                "rot_matrix": eye,
                "pos": [0.0, 0.0, 0.0],
                "is_grounded": True,
                "color_id": 15,
                "temp_support_tag": "",
                "gravity_unit": [0.0, 0.0, -1.0],
                "subtrace": [],
            },
        ),
        matched_port_pairs=(),
        joint_assumptions={
            "clutch_force_capacity_grams": 99.9,
            "moment_arm_capacity_ldu": 40.0,
            "coulomb_friction_coeff": 0.40,
            "normal_crush_ratio": 500.0,
        },
        tolerances={
            "eq_residual_tol_g": 1e-4,
            "ub_violation_tol": 1e-4,
            "fine_csg_voxel_ldu": 2.0,
        },
        lp_witness_summary={"peak_joint_utilization": 0.02},
        cached_dependencies=(),
        claimed_certificates={
            "geometry": "PASS",
            "static_mechanics": "PASS",
            "prefix_mechanics": "PASS",
            "insertion": "PASS",
            "inventory": "PASS",
        },
    )


def test_s1_s2_checkpoint_authenticated_weights_only_and_python_O(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S2: PyTorch security version floor
    for bad_ver in ("2.4.0", "2.5.1+cu121", "2.9.1"):
        with pytest.raises(UnsupportedTorchSecurityVersionError):
            verify_safe_torch_version(bad_ver)
    assert verify_safe_torch_version("2.10.0+cpu") == (2, 10, 0)
    assert verify_safe_torch_version("2.15.0") == (2, 15, 0)

    # S1: Private 0o700 per-run checkpoint directory
    run_dir = create_private_checkpoint_dir(tmp_path / "ckpts", run_id="test_run_01")
    assert (os.lstat(run_dir).st_mode & 0o777) == 0o700
    with pytest.raises(CheckpointSecurityError):
        create_private_checkpoint_dir(tmp_path / "ckpts", run_id="../escape")

    # S1: Save & authenticated CPU reload round-trip of Phase C checkpoint
    torch.manual_seed(7)
    dev = torch.device("cpu")
    enc = TopographicalIPAEncoder(8, 8, 32, 1, 4, 96).to(dev).eval()
    orig_agent = AblationConfigAgent("Config_C", 7, enc, None)
    ckpt_path = run_dir / "Config_C_seed7.pt"
    sha_hex, sz_bytes = safe_save_checkpoint(
        {
            "config_name": "Config_C",
            "seed": np.int64(7),
            "d_model": 32,
            "num_layers": 1,
            "n_heads": 4,
            "encoder_state_dict": enc.state_dict(),
            "planner_state_dict": None,
            "external_10k_solve_accuracy": np.float64(0.42),
        },
        ckpt_path,
    )
    assert sz_bytes > 0
    assert (os.lstat(ckpt_path).st_mode & 0o777) == 0o600

    reloaded_agent, meta = reload_phase_c_checkpoint_agent(
        ckpt_path=ckpt_path,
        expected_sha256=sha_hex,
        device=dev,
    )
    assert meta["sha256"] == sha_hex
    st = Standard8x8State.initial()
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    assert orig_agent.select_moves_batch(
        [st], rng=rng_a, device=dev
    ) == reloaded_agent.select_moves_batch([st], rng=rng_b, device=dev)

    # S1: Tampered expected_sha256 MUST abort BEFORE torch.load is called
    load_calls = 0
    real_torch_load = torch.load

    def _counting_load(*args: object, **kwargs: object) -> object:
        nonlocal load_calls
        load_calls += 1
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", _counting_load)
    wrong_sha = ("0" if sha_hex[0] != "0" else "1") + sha_hex[1:]
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        safe_load_authenticated_checkpoint(ckpt_path, expected_sha256=wrong_sha)
    assert load_calls == 0
    monkeypatch.setattr(torch, "load", real_torch_load)

    # S1: Digest enforcement MUST remain active under `python -O` (no `assert`)
    opt_script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT / "src")!r})
        if __debug__:
            sys.exit(3)
        from topo_chess.security.checkpoint_io import (
            CheckpointIntegrityError,
            safe_load_authenticated_checkpoint,
        )
        try:
            safe_load_authenticated_checkpoint({str(ckpt_path)!r}, expected_sha256={wrong_sha!r})
        except CheckpointIntegrityError:
            sys.exit(0)
        sys.exit(2)
        """
    )
    opt_check = subprocess.run(
        [sys.executable, "-O", "-c", opt_script],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert opt_check.returncode == 0, opt_check.stderr

    # S1: Crafted pickle payload with os.system & VALID SHA-256 MUST be rejected by weights_only=True
    class _ExploitGadget:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return (os.system, ("echo SHOULD_NEVER_EXECUTE",))

    evil_path = run_dir / "evil_gadget.pt"
    buf = io.BytesIO()
    pickle.dump({"gadget": _ExploitGadget()}, buf)
    evil_bytes = buf.getvalue()
    evil_path.write_bytes(evil_bytes)
    evil_sha = hashlib.sha256(evil_bytes).hexdigest()
    with pytest.raises(CheckpointSecurityError, match="weights_only=True"):
        safe_load_authenticated_checkpoint(evil_path, expected_sha256=evil_sha)

    # S1: Oversized model dimension rejected BEFORE nn.Module allocation
    oversized_path = run_dir / "oversized_dims.pt"
    over_sha, _ = safe_save_checkpoint(
        {
            "config_name": "Config_C",
            "seed": 1,
            "d_model": 65536,
            "num_layers": 1,
            "n_heads": 4,
            "encoder_state_dict": {"w": torch.zeros((4, 4))},
            "planner_state_dict": None,
        },
        oversized_path,
    )
    loaded_over, _ = safe_load_authenticated_checkpoint(oversized_path, expected_sha256=over_sha)
    with pytest.raises(CheckpointSchemaError, match="d_model"):
        validate_phase_c_checkpoint_schema(loaded_over)


def test_s3_target_overlap_obligation_never_skipped() -> None:
    prim_3001 = get_brick_primitive("3001")
    matching_boxes = [b.transform(LDUFrame(0, (0, 0, 0))) for b in prim_3001.hull_boxes] + [
        b.transform(LDUFrame(0, (0, 0, 24))) for b in prim_3001.hull_boxes
    ]
    matching_target = TargetShapeBoxes(matching_boxes)
    matching_sha = compute_target_sha256(matching_target)
    art_match = _make_valid_two_brick_cubic_artifact(target_sha256=matching_sha)

    rep_ok = verify_certificate_independently(
        art_match,
        VerificationObligations(expected_target_field=matching_target, min_target_iou=0.95),
    )
    assert rep_ok.verified is True
    assert rep_ok.target_verified is True
    assert rep_ok.target_digest_verified is True
    assert rep_ok.target_overlap_verified is True
    assert rep_ok.measured_target_iou is not None and rep_ok.measured_target_iou >= 0.99

    # Disjoint target field with forged matching target_sha256 MUST fail min_target_iou
    disjoint_target = TargetShapeBoxes([LDUBox(200, 200, 0, 280, 240, 48)])
    disjoint_sha = compute_target_sha256(disjoint_target)
    art_forged_digest = _make_valid_two_brick_cubic_artifact(target_sha256=disjoint_sha)
    rep_low_iou = verify_certificate_independently(
        art_forged_digest,
        VerificationObligations(expected_target_field=disjoint_target, min_target_iou=0.50),
    )
    assert rep_low_iou.target_digest_verified is True
    assert rep_low_iou.target_overlap_verified is False
    assert rep_low_iou.target_verified is False
    assert rep_low_iou.verified is False
    assert any(r.startswith("target_iou_below_minimum:") for r in rep_low_iou.failure_reasons)

    # continuous_se3 + min_target_iou MUST fail closed with explicit unsupported_obligation
    se3_art = _make_valid_single_brick_se3_artifact(target_sha256=matching_sha)
    rep_se3 = verify_certificate_independently(
        se3_art,
        VerificationObligations(expected_target_field=matching_target, min_target_iou=0.25),
    )
    assert rep_se3.verified is False
    assert rep_se3.target_verified is False
    assert rep_se3.target_overlap_verified is False
    assert (
        "unsupported_obligation:min_target_iou_unsupported_for_assembly_mode:continuous_se3"
        in rep_se3.failure_reasons
    )

    # Digest-only + min_target_iou (missing expected_target_field) MUST fail closed
    rep_digest_only = verify_certificate_independently(
        art_match,
        VerificationObligations(expected_target_sha256=matching_sha, min_target_iou=0.50),
    )
    assert rep_digest_only.verified is False
    assert rep_digest_only.target_verified is False
    assert rep_digest_only.target_overlap_verified is False
    assert "invalid_obligation:min_target_iou_requires_expected_target_field" in rep_digest_only.failure_reasons

    # Bare min_target_iou MUST fail closed
    rep_bare = verify_certificate_independently(
        art_match,
        VerificationObligations(min_target_iou=0.50),
    )
    assert rep_bare.verified is False
    assert rep_bare.target_overlap_verified is False

    # Out-of-range min_target_iou MUST fail closed
    rep_nan = verify_certificate_independently(
        art_match,
        VerificationObligations(expected_target_field=matching_target, min_target_iou=1.5),
    )
    assert rep_nan.verified is False
    assert "invalid_obligation:min_target_iou_out_of_range" in rep_nan.failure_reasons


def test_s4_relying_party_inventory_and_mechanics_profile(tmp_path: Path) -> None:
    art_no_inv = _make_valid_two_brick_cubic_artifact(inventory_spec=None)

    # Absent inventory is surfaced explicitly as unchecked_absent unless require_explicit_inventory_spec=True
    rep_default = verify_certificate_independently(art_no_inv)
    assert rep_default.verified is True
    assert rep_default.inventory_status == "unchecked_absent"
    assert len(rep_default.mechanics_profile_sha256) == 64

    rep_req_inv = verify_certificate_independently(
        art_no_inv,
        VerificationObligations(require_explicit_inventory_spec=True),
    )
    assert rep_req_inv.verified is False
    assert rep_req_inv.inventory_status == "failed"
    assert "missing_inventory_spec" in rep_req_inv.failure_reasons

    # Submitter-forged generous inventory_spec={"3001": 99} MUST fail relying-party expected_inventory_counts={"3001": 1}
    art_generous = _make_valid_two_brick_cubic_artifact(inventory_spec={"3001": 99})
    rep_cap = verify_certificate_independently(
        art_generous,
        VerificationObligations(expected_inventory_counts={"3001": 1}),
    )
    assert rep_cap.verified is False
    assert rep_cap.inventory_status == "failed"
    assert "expected_inventory_exceeded:3001" in rep_cap.failure_reasons

    rep_cap_ok = verify_certificate_independently(
        art_generous,
        VerificationObligations(expected_inventory_counts={"3001": 2}),
    )
    assert rep_cap_ok.verified is True
    assert rep_cap_ok.inventory_status == "verified_relying_party"

    # Relying-party foundation policy mismatch MUST fail closed
    rep_fp = verify_certificate_independently(
        art_no_inv,
        VerificationObligations(expected_foundation_policy="table_friction_only"),
    )
    assert rep_fp.verified is False
    assert any(r.startswith("foundation_policy_mismatch:") for r in rep_fp.failure_reasons)

    # Inflated submitter mechanics parameters & tolerances MUST fail MechanicsVerificationProfile bounds
    art_inflated_clutch = _make_valid_two_brick_cubic_artifact()
    d_bad_j = art_inflated_clutch.to_json_dict()
    d_bad_j["joint_assumptions"]["clutch_force_capacity_grams"] = 1.0e9
    rep_bad_j = verify_certificate_independently(VersionedCertificateArtifact.from_json_dict(d_bad_j))
    assert rep_bad_j.verified is False
    assert any("clutch_force_capacity_grams" in r for r in rep_bad_j.failure_reasons)

    d_bad_tol = art_inflated_clutch.to_json_dict()
    d_bad_tol["tolerances"]["eq_residual_tol_g"] = 1.0
    rep_bad_t = verify_certificate_independently(VersionedCertificateArtifact.from_json_dict(d_bad_tol))
    assert rep_bad_t.verified is False
    assert any("eq_residual_tol_g" in r for r in rep_bad_t.failure_reasons)

    # Relying-party MechanicsVerificationProfile override replaces submitter tolerance cleanly
    rep_override = verify_certificate_independently(
        VersionedCertificateArtifact.from_json_dict(d_bad_tol),
        VerificationObligations(
            mechanics_profile=MechanicsVerificationProfile(
                override_tolerances={"eq_residual_tol_g": 1e-4}
            )
        ),
    )
    assert rep_override.verified is True

    # CLI flags enforce relying-party inventory & foundation policy
    cert_json = tmp_path / "cert.json"
    art_no_inv.save_json(cert_json)
    assert verifier_cli_main([str(cert_json), "--require-explicit-inventory"]) == 1
    inv_json = tmp_path / "inv.json"
    inv_json.write_text(json.dumps({"3001": 2}), encoding="utf-8")
    assert verifier_cli_main([str(cert_json), "--expected-inventory-json", str(inv_json)]) == 0


def test_s5_verifier_compiler_and_csv_resource_budgets(tmp_path: Path) -> None:
    art = _make_valid_two_brick_cubic_artifact()
    cert_path = tmp_path / "valid_cert.json"
    art.save_json(cert_path)

    # Pre-read JSON byte limit
    rep_bytes = verify_certificate_json_file(
        cert_path,
        VerificationObligations(resource_limits=VerifierResourceLimits(max_json_bytes=128)),
    )
    assert rep_bytes.verified is False
    assert any("max_json_bytes=128" in r for r in rep_bytes.failure_reasons)

    # Placed-parts cardinality ceiling
    rep_parts = verify_certificate_independently(
        art,
        VerificationObligations(resource_limits=VerifierResourceLimits(max_placed_parts=1)),
    )
    assert rep_parts.verified is False
    assert "resource_limit_exceeded:placed_parts>1" in rep_parts.failure_reasons

    # Isolated subprocess worker succeeds on valid input
    rep_worker = verify_certificate_in_isolated_worker(cert_path, timeout_sec=20.0)
    assert rep_worker.verified is True

    # LDCad grid & DAG expansion budgets fail closed before memory blowup
    with pytest.raises(CatalogResourceLimitError):
        _expand_ldcad_grid("C 1000 C 1000 20 20")
    with pytest.raises(CatalogResourceLimitError):
        _expand_ldcad_grid("C 30 C 30 C 30 10 10 10")
    assert len(_expand_ldcad_grid("C 4 C 2 20 20")) == 8

    budget = _ExpansionBudget(max_nodes=5, max_triangles=10, max_snaps=4)
    for _ in range(5):
        budget.record_node()
    with pytest.raises(CatalogResourceLimitError, match="max_nodes"):
        budget.record_node()

    # Lichess CSV line-byte ceiling & target_total_states validation
    long_stream = io.StringIO("PuzzleId,FEN,Moves\n" + ("A" * 70_000))
    _ = _read_bounded_csv_line(long_stream)
    with pytest.raises(ValueError, match="exceeds maximum byte limit"):
        _read_bounded_csv_line(long_stream)

    chunk_stream = io.StringIO(
        "PuzzleId,FEN,Moves\n" + "".join(f"p{i},fen,e2e4\n" for i in range(5))
    )
    chunks = list(_iter_bounded_csv_chunks(chunk_stream, rows_to_read=5, chunk_sz=2))
    assert [len(c) for c in chunks] == [2, 2, 1]

    with pytest.raises(ValueError, match="target_total_states"):
        build_or_load_phase_c_lichess_datasets(
            csv_path=tmp_path / "missing.csv",
            cache_dir=tmp_path / "cache",
            target_total_states=10_000_000,
        )


def _write_uci_stub(path: Path, body: str) -> Path:
    script = "#!" + sys.executable + "\n" + textwrap.dedent(body).lstrip()
    path.write_text(script, encoding="utf-8")
    path.chmod(0o700)
    return path


def test_s6_stockfish_distiller_stub_eof_hang_and_env_sanitization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    st = Standard8x8State.initial()

    # (a) Premature EOF before uciok -> StockfishProtocolError
    eof_uci = _write_uci_stub(
        tmp_path / "stub_eof_uci.py",
        """
        import sys
        sys.exit(0)
        """,
    )
    with StockfishDistiller(binary_path=str(eof_uci), exchange_timeout_sec=1.0) as dist_eof1:
        with pytest.raises(StockfishProtocolError, match="EOF"):
            dist_eof1.evaluate_state(st)
        assert dist_eof1._proc is None
        assert dist_eof1.engine_failure_count == 1

    # (b) Premature EOF during `go` -> hard StockfishProtocolError (zero silent static fallback!)
    eof_go = _write_uci_stub(
        tmp_path / "stub_eof_go.py",
        """
        import sys
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "uci":
                print("id name StubEngine", flush=True)
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd.startswith("go "):
                sys.exit(1)
        """,
    )
    with StockfishDistiller(binary_path=str(eof_go), exchange_timeout_sec=1.0) as dist_eof2:
        with pytest.raises(StockfishProtocolError, match="EOF"):
            dist_eof2.evaluate_state(st)
        assert dist_eof2._proc is None
        assert dist_eof2.telemetry_summary()["static_fallback_move_count"] >= 0
        assert dist_eof2.telemetry_summary()["engine_failure_count"] == 1

    # (c) Hang past exchange_timeout_sec -> killed & reaped cleanly
    hang_stub = _write_uci_stub(
        tmp_path / "stub_hang.py",
        """
        import sys, time
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "uci":
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd.startswith("go "):
                time.sleep(10.0)
        """,
    )
    with StockfishDistiller(binary_path=str(hang_stub), exchange_timeout_sec=0.25) as dist_hang:
        with pytest.raises(StockfishProtocolError, match="timed out"):
            dist_hang.evaluate_state(st)
        assert dist_hang._proc is None

    # (d) Child environment sanitization + line-count cap enforcement
    monkeypatch.setenv("TOPO_SECRET_CANARY_TOKEN", "TOP_SECRET_DO_NOT_LEAK")
    env_and_spam_stub = _write_uci_stub(
        tmp_path / "stub_env_spam.py",
        """
        import os, sys
        call_idx = 0
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "uci":
                if "TOPO_SECRET_CANARY_TOKEN" in os.environ:
                    sys.exit(99)
                print("uciok", flush=True)
            elif cmd == "isready":
                print("readyok", flush=True)
            elif cmd.startswith("go "):
                call_idx += 1
                if call_idx == 1:
                    print("info depth 1 multipv 1 score cp 35 pv e2e4", flush=True)
                    print("bestmove e2e4", flush=True)
                else:
                    for i in range(60):
                        print(f"info string spam_{i}", flush=True)
                    print("bestmove e2e4", flush=True)
            elif cmd == "quit":
                sys.exit(0)
        """,
    )
    with StockfishDistiller(
        binary_path=str(env_and_spam_stub),
        exchange_timeout_sec=2.0,
        max_exchange_lines=25,
    ) as dist_env:
        q_dict, wdlp = dist_env.evaluate_state(st)
        assert len(q_dict) == 20 and len(wdlp) == 4
        assert dist_env.evaluated_positions_count == 1
        with pytest.raises(StockfishProtocolError, match="exceeded"):
            dist_env.evaluate_state(st)


def test_s7_to_s10_manifest_counts_catalog_provenance_ci_and_dockerfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # S7: WP5 clean-room checkpoint auditor reports separate provenance counts & honors trusted_manifest
    import importlib.util

    wp5_spec = importlib.util.spec_from_file_location(
        "run_wp5_clean_room_repro", REPO_ROOT / "scripts" / "run_wp5_clean_room_repro.py"
    )
    assert wp5_spec is not None and wp5_spec.loader is not None
    wp5_mod = importlib.util.module_from_spec(wp5_spec)
    wp5_spec.loader.exec_module(wp5_mod)

    synth_ckpt_dir = tmp_path / "wp5_ckpts"
    wp4_sub = synth_ckpt_dir / "wp4"
    wp4_sub.mkdir(parents=True, exist_ok=True)
    for cfg_n in ("Config_A", "Config_B", "Config_C", "Config_D"):
        payload_b: dict[str, object] = {"encoder": {"w": torch.ones((2, 2))}}
        if cfg_n == "Config_D":
            payload_b["d3pm_planner"] = {"p": torch.zeros((2, 2))}
        safe_save_checkpoint(payload_b, wp4_sub / f"{cfg_n.lower()}_seed42.pt")

    _, local_manifest, ck_audit = wp5_mod.audit_and_load_wp5_checkpoints(
        synth_ckpt_dir,
        trusted_manifest=None,
        require_trusted_manifest=False,
        seeds=(42,),
    )
    assert ck_audit["verification_mode"] == "self_hashed_local_only"
    assert ck_audit["files_hashed_count"] == 4
    assert ck_audit["files_matched_trusted_manifest_count"] == 0
    assert ck_audit["files_reloaded_weights_only_count"] == 4

    with pytest.raises(CheckpointIntegrityError):
        wp5_mod.audit_and_load_wp5_checkpoints(
            synth_ckpt_dir,
            trusted_manifest=None,
            require_trusted_manifest=True,
            seeds=(42,),
        )

    # CLI --require-trusted-manifest without --expected-manifest MUST fail closed
    res_req_fail = subprocess.run(
        [sys.executable, "scripts/run_wp5_clean_room_repro.py", "--require-trusted-manifest"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_req_fail.returncode != 0

    # Valid trusted_manifest succeeds with authenticated_expected_manifest
    _, _, auth_audit = wp5_mod.audit_and_load_wp5_checkpoints(
        synth_ckpt_dir,
        trusted_manifest=local_manifest,
        require_trusted_manifest=True,
        seeds=(42,),
    )
    assert auth_audit["verification_mode"] == "authenticated_expected_manifest"
    assert auth_audit["files_matched_trusted_manifest_count"] == 4

    # Tampered trusted_manifest MUST fail closed BEFORE loading
    bad_manifest = {k: ("f" * 64) for k in local_manifest}
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        wp5_mod.audit_and_load_wp5_checkpoints(
            synth_ckpt_dir,
            trusted_manifest=bad_manifest,
            require_trusted_manifest=True,
            seeds=(42,),
        )

    # S8: Bundled release asset digests match pinned release provenance
    verified_assets = verify_bundled_asset_digests()
    assert set(verified_assets.keys()) == set(PINNED_RELEASE_ASSET_DIGESTS_V5_2.keys())

    # S8: CompiledLDrawCatalog fails closed when manifest is missing or omits catalog_gz_sha256
    fake_gz = tmp_path / "compiled_ldraw_catalog.json.gz"
    with gzip.open(fake_gz, "wt", encoding="utf-8") as gf:
        json.dump({"parts": {}, "aliases": {}}, gf)
    fake_gz_sha = hashlib.sha256(fake_gz.read_bytes()).hexdigest()

    cat_missing_manifest = CompiledLDrawCatalog(
        catalog_gz_path=fake_gz,
        coacd_npz_path=tmp_path / "no_coacd.npz",
        manifest_path=tmp_path / "missing_manifest.json",
    )
    with pytest.raises(CatalogProvenanceError, match="Missing required compiled catalog manifest"):
        cat_missing_manifest._ensure_loaded()

    incomplete_manifest = tmp_path / "incomplete_manifest.json"
    incomplete_manifest.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")
    cat_incomplete = CompiledLDrawCatalog(
        catalog_gz_path=fake_gz,
        coacd_npz_path=tmp_path / "no_coacd.npz",
        manifest_path=incomplete_manifest,
    )
    with pytest.raises(CatalogProvenanceError, match="catalog_gz_sha256"):
        cat_incomplete._ensure_loaded()

    no_parts_manifest = tmp_path / "no_parts_manifest.json"
    no_parts_manifest.write_text(
        json.dumps({"schema_version": "1.0.0", "catalog_gz_sha256": fake_gz_sha}),
        encoding="utf-8",
    )
    cat_no_parts = CompiledLDrawCatalog(
        catalog_gz_path=fake_gz,
        coacd_npz_path=tmp_path / "no_coacd.npz",
        manifest_path=no_parts_manifest,
    )
    with pytest.raises(CatalogProvenanceError, match="total_compiled_parts"):
        cat_no_parts._ensure_loaded()

    # S8: Joint replacement of (catalog + manifest) without trusted_custom_catalog_sha256 fails closed
    monkeypatch.setattr(
        "topo_brick.evaluation.independent_verifier.PINNED_RELEASE_CATALOG_SHA256_BY_SCHEMA",
        {CERTIFICATE_SCHEMA_VERSION: "0" * 64},
    )
    art = _make_valid_two_brick_cubic_artifact()
    rep_unpinned = verify_certificate_independently(art)
    assert rep_unpinned.verified is False
    assert rep_unpinned.catalog_sha256_valid is False
    assert "local_catalog_differs_from_pinned_release" in rep_unpinned.failure_reasons

    # Explicit trusted_custom_catalog_sha256 allows audited custom catalog builds
    rep_custom_ok = verify_certificate_independently(
        art,
        VerificationObligations(trusted_custom_catalog_sha256=compute_catalog_sha256()),
    )
    assert rep_custom_ok.verified is True
    assert rep_custom_ok.catalog_sha256_valid is True

    # S9 & S10: Supply-chain, GitHub Actions, lockfile, and Dockerfile static security audit
    pyproject_txt = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "5.2.1"' in pyproject_txt
    assert '"torch>=2.10.0"' in pyproject_txt
    assert (REPO_ROOT / "SECURITY.md").is_file()

    for lock_rel in ("requirements/build-lock.txt", "requirements/py313-lock.txt"):
        lock_txt = (REPO_ROOT / lock_rel).read_text(encoding="utf-8")
        pkg_lines = [
            ln.strip()
            for ln in lock_txt.splitlines()
            if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("--hash=")
        ]
        assert len(pkg_lines) >= 2
        assert "--hash=sha256:" in lock_txt
        for digest_m in re.findall(r"--hash=sha256:([0-9a-f]+)", lock_txt):
            assert len(digest_m) == 64

    sha_pin_re = re.compile(r"^[^@]+@[0-9a-f]{40}$")
    for wf_rel in (".github/workflows/ci.yml", ".github/workflows/full-repro.yml"):
        wf_data = yaml.safe_load((REPO_ROOT / wf_rel).read_text(encoding="utf-8"))
        assert wf_data.get("permissions") == {"contents": "read"}, wf_rel
        for job in wf_data.get("jobs", {}).values():
            for step in job.get("steps", []):
                if "uses" in step:
                    uses_val = str(step["uses"]).strip()
                    assert sha_pin_re.match(uses_val), f"Unpinned action in {wf_rel}: {uses_val}"
                    if uses_val.startswith("actions/checkout@"):
                        assert step.get("with", {}).get("persist-credentials") is False

    dockerfile_txt = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(
        r"^FROM python:3\.13-slim-bookworm@sha256:[0-9a-f]{64}$",
        dockerfile_txt,
        flags=re.MULTILINE,
    )
    assert "PYTHONHASHSEED=42" in dockerfile_txt
    assert "--require-hashes" in dockerfile_txt
    assert "--no-build-isolation" in dockerfile_txt
    assert "chmod -R a-w /workspace/topographical-tensors" in dockerfile_txt
    assert re.search(r"^USER 10001:10001$", dockerfile_txt, flags=re.MULTILINE)
