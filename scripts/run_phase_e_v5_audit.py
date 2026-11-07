#!/usr/bin/env python3
"""CLI entrypoint for Phase E (v5.0.0) Round-2 42-Item Audit & Deconfounded Control Suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from topo_chess.evaluation.phase_e_round2_runner import (
    generate_manuscript_tables_v5_markdown,
    run_phase_e_v5_audit,
)


def main() -> int:
    import os

    default_data_dir = Path(os.environ.get("TOPO_DATA_DIR", str(ROOT / "data")))
    parser = argparse.ArgumentParser(description="Run Phase E (v5.0.0) Round-2 Audit & Control Suite")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_data_dir / "raw" / "lichess_db_puzzle.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_data_dir,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "wp4_phase_e",
    )
    parser.add_argument(
        "--phase-d-report",
        type=Path,
        default=ROOT / "logs" / "phase_d_v4_audit_report.json",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "logs" / "phase_e_v5_audit_report.json",
    )
    parser.add_argument(
        "--tables-out",
        type=Path,
        default=ROOT / "evaluations" / "manuscript_tables_v5.0_phase_e.md",
    )
    parser.add_argument("--encoder-steps", type=int, default=15)
    args = parser.parse_args()

    report = run_phase_e_v5_audit(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        phase_d_report_path=args.phase_d_report,
        encoder_steps=args.encoder_steps,
    )

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_tables = generate_manuscript_tables_v5_markdown(report)
    args.tables_out.parent.mkdir(parents=True, exist_ok=True)
    args.tables_out.write_text(md_tables, encoding="utf-8")

    print(
        f"[Phase E v5.0.0 Complete] all_42_round2_items_resolved={report['all_42_round2_items_resolved']} in {report['total_wall_clock_sec']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
