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

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, Tuple
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


def main() -> None:
    t0 = time.perf_counter()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 84)
    print(f"TOPOGRAPHICAL TENSORS — WP5 CLEAN-ROOM REPRODUCIBILITY & EDITORIAL AUDIT ({device})")
    print("=" * 84)

    import os

    data_root = Path(os.environ.get("TOPO_DATA_DIR", "./data"))
    log_dir = data_root / "logs"
    ckpt_dir = data_root / "checkpoints"
    wp4_ckpt_dir = ckpt_dir / "wp4"
    eval_dir = data_root / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    sg123_path = log_dir / "stage_gates_1_2_3_report.json"
    sg4_path = log_dir / "stage_gate_4_report.json"
    assert sg123_path.is_file(), f"Missing {sg123_path}"
    assert sg4_path.is_file(), f"Missing {sg4_path}"

    sg123_report = json.loads(sg123_path.read_text(encoding="utf-8"))
    sg4_report = json.loads(sg4_path.read_text(encoding="utf-8"))

    # 1. Compute SHA-256 manifest of all saved checkpoints
    sha256_manifest: Dict[str, str] = {}
    wp2_ckpt = ckpt_dir / "wp2_ipa_5x5_verified.pt"
    if wp2_ckpt.is_file():
        sha256_manifest[wp2_ckpt.name] = compute_file_sha256(wp2_ckpt)

    seeds = (42, 1337, 2026)
    ckpt_bundles: Dict[Tuple[str, int], Dict[str, Dict[str, torch.Tensor]]] = {}
    for cfg_name in ("Config_A", "Config_B", "Config_C", "Config_D"):
        for seed in seeds:
            fname = f"{cfg_name.lower()}_seed{seed}.pt"
            fpath = wp4_ckpt_dir / fname
            assert fpath.is_file(), f"Missing WP4 checkpoint: {fpath}"
            sha256_manifest[f"wp4/{fname}"] = compute_file_sha256(fpath)
            ckpt_bundles[(cfg_name, seed)] = torch.load(
                fpath, map_location=device, weights_only=True
            )

    print(f"\n[WP5 Checkpoint Integrity Manifest: {len(sha256_manifest)} Checkpoints Verified]")
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
    repro_report_path = log_dir / "stage_gate_5_repro_report.json"
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
