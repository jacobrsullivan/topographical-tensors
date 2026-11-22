#!/usr/bin/env python3
"""End-to-End WP5 Clean-Room Reproducibility & Editorial Audit Runner (`v2.0`).

Executes Stage-Gate 5 verification criteria:
1. SG5-A (Clean-Room Reproduction within +/-0.5%):
   - Computes SHA-256 cryptographic hashes of all 13 saved checkpoints.
   - Instantiates brand-new models in a clean room, loads saved `state_dict` weights from
     `./data/checkpoints/wp4/`, and regenerates all accuracy,
     legality, Kendall tau, and BayesElo metrics.
   - Verifies maximum relative deviation across all reported metrics is <= 0.5%.
2. SG5-B (NeurIPS Reproducibility Checklist, ASD-STE100 & Risk R4 Dual-Benchmark Audit):
   - Verifies 100% completion of formal mathematical proofs, hyperparameter/compute logs,
     dataset datasheet, Risk R4 dual CPU/GPU throughput reporting, Dockerfile, and NeurIPS/STE100 checklists.
3. Regenerates publication-ready Markdown manuscript tables at
   `./data/evaluations/manuscript_tables_v2.0.md`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from topo_chess.data.pipeline import (
    build_gardner_mate_in_3_benchmark,
    generate_synthetic_gardner_games,
    partition_and_deduplicate_games,
)
from topo_chess.domains.boards import Gardner5x5State
from topo_chess.evaluation.clean_room_repro import (
    StageGate5Report,
    audit_editorial_and_neurips_compliance,
    compute_file_sha256,
    generate_manuscript_markdown_tables,
    verify_clean_room_reproduction_from_checkpoints,
)
from topo_chess.evaluation.gauntlet_and_bayeselo import BayesEloEntry, MatchResult
from topo_chess.security.checkpoint_io import (
    CheckpointIntegrityError,
    safe_load_authenticated_checkpoint,
    validate_wp4_state_bundle,
)


def audit_and_load_wp5_checkpoints(
    ckpt_dir: Path,
    *,
    trusted_manifest: Optional[Dict[str, str]] = None,
    require_trusted_manifest: bool = False,
    device: torch.device = torch.device("cpu"),
    seeds: Tuple[int, ...] = (42, 1337, 2026),
) -> Tuple[
    Dict[Tuple[str, int], Dict[str, Dict[str, torch.Tensor]]],
    Dict[str, str],
    Dict[str, Any],
]:
    """
    Authenticate and load WP5 clean-room checkpoints (`S1`, `S2`, `S7`), returning
    `(ckpt_bundles, sha256_manifest, checkpoint_audit)` with distinct provenance counts.
    """
    if require_trusted_manifest and trusted_manifest is None:
        raise CheckpointIntegrityError(
            "require_trusted_manifest=True, but no trusted_manifest was supplied."
        )
    wp4_ckpt_dir = ckpt_dir / "wp4"
    sha256_manifest: Dict[str, str] = {}
    file_audit_records: List[Dict[str, Any]] = []
    files_hashed_count = 0
    files_matched_trusted_count = 0
    files_reloaded_count = 0

    wp2_ckpt = ckpt_dir / "wp2_ipa_5x5_verified.pt"
    if wp2_ckpt.is_file():
        exp_wp2 = trusted_manifest.get(wp2_ckpt.name) if trusted_manifest is not None else None
        if require_trusted_manifest and exp_wp2 is None:
            raise CheckpointIntegrityError(f"Missing {wp2_ckpt.name} in trusted manifest")
        wp2_digest = compute_file_sha256(wp2_ckpt)
        if exp_wp2 is not None and wp2_digest != exp_wp2.strip().lower():
            raise CheckpointIntegrityError(
                f"SHA-256 mismatch on {wp2_ckpt.name}: expected {exp_wp2}, got {wp2_digest}"
            )
        sha256_manifest[wp2_ckpt.name] = wp2_digest
        files_hashed_count += 1
        if exp_wp2 is not None:
            files_matched_trusted_count += 1
        file_audit_records.append(
            {
                "file": wp2_ckpt.name,
                "sha256": wp2_digest,
                "matched_trusted_manifest": exp_wp2 is not None,
                "reloaded_weights_only": False,
                "repro_eval_checked": False,
            }
        )

    ckpt_bundles: Dict[Tuple[str, int], Dict[str, Dict[str, torch.Tensor]]] = {}
    for cfg_name in ("Config_A", "Config_B", "Config_C", "Config_D"):
        for seed in seeds:
            fname = f"{cfg_name.lower()}_seed{seed}.pt"
            rel_key = f"wp4/{fname}"
            fpath = wp4_ckpt_dir / fname
            if not fpath.is_file():
                raise FileNotFoundError(f"Missing WP4 checkpoint: {fpath}")
            exp_sha = trusted_manifest.get(rel_key) if trusted_manifest is not None else None
            if require_trusted_manifest and exp_sha is None:
                raise CheckpointIntegrityError(f"Missing {rel_key} in trusted manifest")
            raw_bundle, actual_sha = safe_load_authenticated_checkpoint(
                fpath,
                expected_sha256=exp_sha,
                map_location="cpu",
            )
            validated_bundle = validate_wp4_state_bundle(
                raw_bundle, require_planner=(cfg_name == "Config_D")
            )
            ckpt_bundles[(cfg_name, seed)] = {
                k: {pk: pv.to(device) for pk, pv in sd.items()}
                for k, sd in validated_bundle.items()
            }
            sha256_manifest[rel_key] = actual_sha
            files_hashed_count += 1
            files_reloaded_count += 1
            if exp_sha is not None:
                files_matched_trusted_count += 1
            file_audit_records.append(
                {
                    "file": rel_key,
                    "sha256": actual_sha,
                    "matched_trusted_manifest": exp_sha is not None,
                    "reloaded_weights_only": True,
                    "repro_eval_checked": True,
                }
            )

    checkpoint_audit = {
        "verification_mode": (
            "authenticated_expected_manifest"
            if trusted_manifest is not None
            else "self_hashed_local_only"
        ),
        "trusted_manifest_supplied": trusted_manifest is not None,
        "files_hashed_count": files_hashed_count,
        "files_matched_trusted_manifest_count": files_matched_trusted_count,
        "files_reloaded_weights_only_count": files_reloaded_count,
        "manifest": sha256_manifest,
        "file_checks": file_audit_records,
    }
    return ckpt_bundles, sha256_manifest, checkpoint_audit


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="WP5 Clean-Room Reproducibility & Checkpoint Provenance Runner."
    )
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        default=None,
        help="Optional trusted JSON mapping checkpoint relative paths to expected SHA-256 digests.",
    )
    parser.add_argument(
        "--require-trusted-manifest",
        action="store_true",
        help="Fail closed if --expected-manifest is absent or missing any checkpoint entry.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override path for the Stage-Gate 5 JSON reproduction report.",
    )
    args = parser.parse_args(argv)

    trusted_manifest: Optional[Dict[str, str]] = None
    if args.expected_manifest is not None:
        raw_tm = json.loads(args.expected_manifest.read_text(encoding="utf-8"))
        if not isinstance(raw_tm, dict):
            raise CheckpointIntegrityError("--expected-manifest must be a JSON object")
        trusted_manifest = {str(k): str(v) for k, v in raw_tm.items()}
    elif args.require_trusted_manifest:
        raise CheckpointIntegrityError(
            "--require-trusted-manifest set, but --expected-manifest was not supplied."
        )

    t0 = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 84)
    print(f"TOPOGRAPHICAL TENSORS — WP5 CLEAN-ROOM REPRODUCIBILITY & EDITORIAL AUDIT ({device})")
    print("=" * 84)

    import os

    data_root = Path(os.environ.get("TOPO_DATA_DIR", "./data"))
    log_dir = data_root / "logs"
    ckpt_dir = data_root / "checkpoints"
    eval_dir = data_root / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    sg123_path = log_dir / "stage_gates_1_2_3_report.json"
    sg4_path = log_dir / "stage_gate_4_report.json"
    if not sg123_path.is_file():
        raise FileNotFoundError(f"Missing {sg123_path}")
    if not sg4_path.is_file():
        raise FileNotFoundError(f"Missing {sg4_path}")

    sg123_report = json.loads(sg123_path.read_text(encoding="utf-8"))
    sg4_report = json.loads(sg4_path.read_text(encoding="utf-8"))

    seeds = (42, 1337, 2026)
    ckpt_bundles, sha256_manifest, checkpoint_audit = audit_and_load_wp5_checkpoints(
        ckpt_dir,
        trusted_manifest=trusted_manifest,
        require_trusted_manifest=args.require_trusted_manifest,
        device=device,
        seeds=seeds,
    )
    files_hashed_count = int(checkpoint_audit["files_hashed_count"])
    files_matched_trusted_count = int(checkpoint_audit["files_matched_trusted_manifest_count"])
    files_reloaded_count = int(checkpoint_audit["files_reloaded_weights_only_count"])

    if trusted_manifest is not None:
        header_label = (
            f"[WP5 Checkpoint Manifest: {files_hashed_count} Hashed, "
            f"{files_matched_trusted_count} Matched Trusted Manifest, "
            f"{files_reloaded_count} Reloaded (weights_only=True)]"
        )
    else:
        header_label = (
            f"[WP5 Checkpoint Manifest: {files_hashed_count} Hashed & {files_reloaded_count} Reloaded "
            "(0 Matched Against External Trusted Manifest — Local Self-Hash)]"
        )
    print(f"\n{header_label}")
    for k, digest in sha256_manifest.items():
        print(f"  {k:32s}: sha256:{digest[:16]}...{digest[-16:]}")

    # 2. Reconstruct identical benchmark splits & puzzles in clean room
    games = generate_synthetic_gardner_games(num_games=65, max_plies=16, seed=42)
    split = partition_and_deduplicate_games(games)
    all_records = split.train + split.val + split.test
    test_records = all_records[280:360] if len(all_records) >= 360 else split.test
    puzzles = build_gardner_mate_in_3_benchmark(num_puzzles=100)
    legality_states = [Gardner5x5State.from_fen(r.fen) for r in all_records[:250]]

    recorded_matches = [MatchResult(**m) for m in sg4_report["match_results"]]
    recorded_leaderboard = [BayesEloEntry(**e) for e in sg4_report["bayeselo_leaderboard"]]

    # 3. Execute Clean-Room Reproduction Verification (SG5-A)
    deviations, max_dev, sg5_a_pass = verify_clean_room_reproduction_from_checkpoints(
        checkpoint_bundles=ckpt_bundles,
        recorded_summary=sg4_report["config_summary"],
        recorded_matches=recorded_matches,
        recorded_leaderboard=recorded_leaderboard,
        test_records=test_records,
        puzzles=puzzles,
        legality_states=legality_states,
        seeds=seeds,
        d_model=128,
        num_layers=4,
        n_heads=4,
        d_ff=384,
        tolerance_pct=0.5,
        device=device,
    )

    print("\n[WP5 Clean-Room Metric Reproduction Audit (SG5-A, Tolerance <= +/-0.5%)]")
    for d in deviations:
        print(
            f"  {d.metric_name:44s}: Recorded = {d.recorded_value:9.4f} | "
            f"Reproduced = {d.reproduced_value:9.4f} | Deviation = {d.pct_deviation:6.4f}%"
        )

    # 4. Execute Editorial, NeurIPS Checklist & Risk R4 Audit (SG5-B)
    editorial_audit = audit_editorial_and_neurips_compliance(REPO_ROOT)
    sg5_b_pass = editorial_audit.passed

    print("\n[WP5 Editorial, NeurIPS Checklist & Risk R4 Audit (SG5-B)]")
    print(f"  Formal Mathematical Proofs (Theorems 1-3, Prop 4): {editorial_audit.proofs_complete}")
    print(
        f"  Hyperparameter Table, Compute Log & Datasheet:     {editorial_audit.hyperparameters_and_datasheet_complete}"
    )
    print(
        f"  Risk R4 Dual CPU/GPU Throughput Reporting:         {editorial_audit.risk_r4_dual_benchmark_reported}"
    )
    print(
        f"  NeurIPS Reproducibility Checklist (10/10):         {editorial_audit.neurips_checklist_complete}"
    )
    print(
        f"  ASD-STE100 Editorial Compliance (6/6):             {editorial_audit.ste100_compliance_complete}"
    )
    print(f"  Deterministic Dockerfile Present:                  {editorial_audit.dockerfile_present}")
    print(
        f"  Flight Recorder Complete (ALT-001..007, OPS-001):  {editorial_audit.flight_recorder_complete}"
    )

    sg5_report = StageGate5Report(
        checkpoint_sha256_manifest=sha256_manifest,
        metric_deviations=deviations,
        max_pct_deviation=max_dev,
        tolerance_pct=0.5,
        sg5_a_clean_room_repro_pass=sg5_a_pass,
        editorial_audit=editorial_audit,
        sg5_b_editorial_compliance_pass=sg5_b_pass,
        stage_gate_5_overall_pass=(sg5_a_pass and sg5_b_pass),
    )

    # 5. Regenerate Manuscript Markdown Tables & Save Report
    tables_md = generate_manuscript_markdown_tables(sg123_report, sg4_report, sg5_report)
    tables_path = eval_dir / "manuscript_tables_v2.0.md"
    tables_path.write_text(tables_md, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    payload = asdict(sg5_report)
    payload["wall_clock_sec"] = elapsed
    payload["checkpoint_audit"] = checkpoint_audit
    repro_report_path = (
        args.output if args.output is not None else (log_dir / "stage_gate_5_repro_report.json")
    )
    repro_report_path.parent.mkdir(parents=True, exist_ok=True)
    repro_report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 84)
    print(
        f"STAGE-GATE 5 OVERALL VERDICT: {'PASSED' if sg5_report.stage_gate_5_overall_pass else 'FAILED'} "
        f"(Total Wall-Clock: {elapsed:.2f}s)"
    )
    print(
        f"  SG5-A (Clean-Room Reproduction <= 0.5%): Max Deviation = {max_dev:.4f}% ({sg5_a_pass})"
    )
    print(
        f"  SG5-B (NeurIPS, ASD-STE100 & Risk R4 Audit): {editorial_audit.checklist_items_verified}/"
        f"{editorial_audit.checklist_items_total} Items ({sg5_b_pass})"
    )
    print("=" * 84)
    print(f"Saved Stage-Gate 5 JSON report to {repro_report_path}")
    print(f"Saved regenerated manuscript tables to {tables_path}")

    if not sg5_report.stage_gate_5_overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
