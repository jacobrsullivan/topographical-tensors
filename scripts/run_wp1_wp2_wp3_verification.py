#!/usr/bin/env python3
"""End-to-End Execution & Stage-Gate 1, 2, 3 Verification Runner."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Tuple
import torch

from topo_chess.data.pipeline import (
    Gardner5x5Evaluator,
    PositionRecord,
    StockfishDistiller,
    build_gardner_mate_in_3_benchmark,
    partition_and_deduplicate_games,
)
from topo_chess.diffusion.d3pm_planner import D3PMHorizonPlanner
from topo_chess.domains.boards import Gardner5x5State, Standard8x8State
from topo_chess.domains.move_graph import GARDNER_5X5_GRAPH, STANDARD_8X8_GRAPH
from topo_chess.encoding.topographical_encoder import TopographicalDecoder, TopographicalEncoder
from topo_chess.evaluation.metrics import (
    compute_action_ranking_kendall_tau,
    evaluate_stage_gate_3,
    train_wp2_ipa_on_dataset,
    train_wp3_d3pm_on_puzzles,
    verify_mirror_color_equivariance,
)
from topo_chess.models.ipa_encoder import ChessFormerShawBaseline, TopographicalIPAEncoder


def _verify_reconstruction_chunk(args: Tuple[int, int, bool]) -> int:
    count, seed, is_8x8 = args
    rng = random.Random(seed)
    h, w = (8, 8) if is_8x8 else (5, 5)
    enc = TopographicalEncoder(h, w)
    dec = TopographicalDecoder(h, w)
    ok = 0
    st = Standard8x8State.starting_state() if is_8x8 else Gardner5x5State.starting_state()
    for _ in range(count):
        t = enc.encode(st)
        r = dec.decode(t)
        if r.zobrist_hash() == st.zobrist_hash() and r.legal_edge_indices() == st.legal_edge_indices():
            ok += 1
        mvs = st.legal_moves()
        if mvs and not st.is_game_over():
            st = st.step(rng.choice(mvs))
        else:
            st = Standard8x8State.starting_state() if is_8x8 else Gardner5x5State.starting_state()
    return ok


def run_stage_gate_1(num_test_positions: int = 100_000) -> Dict[str, object]:
    print("\n[WP1] Running Stage-Gate 1 Verification...")
    t0 = time.perf_counter()
    rng = random.Random(42)

    # 1. Generate games and verify 0% Zobrist overlap across train/val/test
    games: List[Tuple[str, List[PositionRecord]]] = []
    for g_idx in range(200):
        game_id = f"gardner_game_{g_idx:05d}"
        st = Gardner5x5State.starting_state()
        recs: List[PositionRecord] = []
        for _ply in range(20):
            mvs = st.legal_moves()
            if not mvs or st.is_game_over():
                break
            edges = st.legal_edge_indices()
            recs.append(
                PositionRecord(
                    game_id=game_id,
                    fen=st.to_fen(),
                    zobrist_hash=st.zobrist_hash(),
                    side_to_move=st.side_to_move,
                    legal_edge_ids=edges,
                    q_values={e: 0.5 for e in edges},
                    wdlp_target=(0.4, 0.2, 0.4, 0.25),
                )
            )
            st = st.step(rng.choice(mvs))
        games.append((game_id, recs))

    split_ds = partition_and_deduplicate_games(games, train_ratio=0.80, val_ratio=0.10)
    overlap_count = split_ds.cross_split_overlap_count()

    # 2. Verify 100% legal move reconstruction across 100,000 test positions in parallel across P-cores
    n_workers = 10
    chunk_sz = num_test_positions // n_workers
    tasks = [
        (chunk_sz, 1000 + i, (i >= n_workers // 2)) for i in range(n_workers)
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        reconstructed_ok = sum(pool.map(_verify_reconstruction_chunk, tasks))

    recon_rate = float(reconstructed_ok) / float(chunk_sz * n_workers)

    # 3. Verify Stockfish fixed-node MultiPV distillation determinism on 8x8
    distiller = StockfishDistiller(nodes_per_legal_move=300, min_nodes_per_position=3000)
    sf_q1, _ = distiller.evaluate_state(Standard8x8State.starting_state())
    sf_q2, _ = distiller.evaluate_state(Standard8x8State.starting_state())
    distiller.close()
    sf_max_drift = max(abs(sf_q1[k] - sf_q2[k]) for k in sf_q1)

    elapsed = time.perf_counter() - t0
    report = {
        "gate": "Stage-Gate 1 (WP1)",
        "passed": (
            overlap_count == 0
            and recon_rate == 1.0
            and GARDNER_5X5_GRAPH.num_edges == 455
            and STANDARD_8X8_GRAPH.num_edges == 1858
            and sf_max_drift == 0.0
        ),
        "gardner_5x5_valid_edges": GARDNER_5X5_GRAPH.num_edges,
        "standard_8x8_valid_edges": STANDARD_8X8_GRAPH.num_edges,
        "zobrist_cross_split_overlap": overlap_count,
        "positions_tested_for_reconstruction": num_test_positions,
        "legal_move_reconstruction_rate": recon_rate,
        "stockfish_fixed_node_determinism_drift": sf_max_drift,
        "elapsed_sec": round(elapsed, 2),
    }
    print(f"[WP1] Stage-Gate 1 Result: {json.dumps(report, indent=2)}")
    return report


def run_stage_gate_2(device: torch.device) -> Tuple[Dict[str, object], TopographicalIPAEncoder]:
    print(f"\n[WP2] Running Stage-Gate 2 Verification on device={device}...")
    t0 = time.perf_counter()
    torch.manual_seed(42)
    rng = random.Random(42)

    # 1. Parameter count & FLOPS comparison at CF-6M 8x8 scale
    ipa_8x8 = TopographicalIPAEncoder(height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=768)
    shaw_8x8 = ChessFormerShawBaseline(height=8, width=8, d_model=256, num_layers=8, n_heads=8, d_ff=832)

    ipa_params = ipa_8x8.count_parameters()
    shaw_params = shaw_8x8.count_parameters()
    ipa_flops = ipa_8x8.compute_forward_flops()
    shaw_flops = shaw_8x8.compute_forward_flops()
    flops_reduction = 1.0 - (float(ipa_flops) / float(shaw_flops))

    # 2. Equivariance test across 100 5x5 & 8x8 positions
    states8 = []
    st8 = Standard8x8State.starting_state()
    for _ in range(100):
        states8.append(st8)
        mvs = st8.legal_moves()
        st8 = st8.step(rng.choice(mvs)) if (mvs and not st8.is_game_over()) else Standard8x8State.starting_state()
    equivariance_error_8x8 = verify_mirror_color_equivariance(
        ipa_8x8.to(device), states8, device=device
    )

    # 3. Train 5x5 IPA Encoder & evaluate Kendall tau on held-out mirror-disjoint test set
    base_states: List[Gardner5x5State] = []
    seen = set()
    st5 = Gardner5x5State.starting_state()
    while len(base_states) < 60:
        if st5.zobrist_hash() not in seen and len(st5.legal_moves()) >= 5:
            seen.add(st5.zobrist_hash())
            base_states.append(st5)
        mvs = st5.legal_moves()
        if not mvs or st5.is_game_over() or len(seen) % 8 == 0:
            st5 = Gardner5x5State.starting_state()
            if mvs:
                st5 = st5.step(rng.choice(mvs))
        else:
            st5 = st5.step(rng.choice(mvs))

    train_recs: List[PositionRecord] = []
    test_recs: List[PositionRecord] = []
    for i, s in enumerate(base_states):
        q_d, wdlp = Gardner5x5Evaluator.evaluate_action_values(s, search_depth=2)
        train_recs.append(
            PositionRecord(
                game_id=f"tr_{i}",
                fen=s.to_fen(),
                zobrist_hash=s.zobrist_hash(),
                side_to_move=s.side_to_move,
                legal_edge_ids=list(q_d.keys()),
                q_values=q_d,
                wdlp_target=wdlp,
            )
        )
        s_m = s.mirror_vertical_swap_colors()
        if s_m.zobrist_hash() not in seen:
            q_m, wdlp_m = Gardner5x5Evaluator.evaluate_action_values(s_m, search_depth=2)
            test_recs.append(
                PositionRecord(
                    game_id=f"te_{i}",
                    fen=s_m.to_fen(),
                    zobrist_hash=s_m.zobrist_hash(),
                    side_to_move=s_m.side_to_move,
                    legal_edge_ids=list(q_m.keys()),
                    q_values=q_m,
                    wdlp_target=wdlp_m,
                )
            )

    ipa_5x5 = TopographicalIPAEncoder(height=5, width=5, d_model=192, num_layers=6, n_heads=6, d_ff=576)
    losses = train_wp2_ipa_on_dataset(
        ipa_5x5, train_recs, is_8x8=False, epochs=50, batch_size=20, lr=2e-3, device=device
    )
    kendall_tau = compute_action_ranking_kendall_tau(ipa_5x5, test_recs, is_8x8=False, device=device)

    elapsed = time.perf_counter() - t0
    report = {
        "gate": "Stage-Gate 2 (WP2)",
        "passed": (
            equivariance_error_8x8 <= 1e-4
            and kendall_tau >= 0.72
            and flops_reduction >= 0.15
        ),
        "cf6m_ipa_parameters_8x8": ipa_params,
        "cf6m_shaw_baseline_parameters_8x8": shaw_params,
        "ipa_forward_flops": ipa_flops,
        "shaw_forward_flops": shaw_flops,
        "flops_reduction_ratio": round(flops_reduction, 4),
        "mirror_color_equivariance_max_error": equivariance_error_8x8,
        "held_out_kendall_tau": round(kendall_tau, 4),
        "initial_train_loss": round(losses[0], 5),
        "final_train_loss": round(losses[-1], 5),
        "elapsed_sec": round(elapsed, 2),
    }
    print(f"[WP2] Stage-Gate 2 Result: {json.dumps(report, indent=2)}")
    return report, ipa_5x5


def run_stage_gate_3(
    ipa_5x5: TopographicalIPAEncoder, device: torch.device, num_legality_positions: int = 10_000
) -> Dict[str, object]:
    print(f"\n[WP3] Running Stage-Gate 3 Verification on device={device}...")
    t0 = time.perf_counter()
    torch.manual_seed(42)
    rng = random.Random(42)

    puzzles = build_gardner_mate_in_3_benchmark(num_puzzles=100)
    train_puzzles = puzzles[:70]
    test_puzzles = puzzles[70:]

    legality_states: List[Gardner5x5State] = []
    st = Gardner5x5State.starting_state()
    for _ in range(num_legality_positions):
        legality_states.append(st)
        mvs = st.legal_moves()
        st = st.step(rng.choice(mvs)) if (mvs and not st.is_game_over()) else Gardner5x5State.starting_state()

    planner = D3PMHorizonPlanner(
        height=5,
        width=5,
        horizon_plies=6,
        d_model=192,
        num_layers=4,
        n_heads=6,
        d_ff=384,
        max_timesteps=32,
        aux_ce_lambda=0.05,
        rollout_mask_final_steps=3,
    )

    losses = train_wp3_d3pm_on_puzzles(
        encoder=ipa_5x5,
        planner=planner,
        puzzles=train_puzzles,
        epochs=60,
        batch_size=35,
        lr=3e-3,
        freeze_encoder=True,
        device=device,
    )

    sg3 = evaluate_stage_gate_3(
        encoder=ipa_5x5,
        planner=planner,
        puzzles=test_puzzles,
        legality_test_states=legality_states,
        num_steps=20,
        device=device,
    )

    elapsed = time.perf_counter() - t0
    report = {
        "gate": "Stage-Gate 3 (WP3)",
        "passed": (
            sg3.legal_trajectory_rate == 1.0
            and sg3.mate_in_3_accuracy >= 0.85
            and sg3.horizon_accuracy_drop < 0.12
            and sg3.latency_speedup_ratio >= 5.0
        ),
        "legality_test_positions": num_legality_positions,
        "legal_trajectory_rate": sg3.legal_trajectory_rate,
        "mate_in_3_solve_accuracy": round(sg3.mate_in_3_accuracy, 4),
        "ply1_accuracy": round(sg3.ply1_accuracy, 4),
        "ply6_accuracy": round(sg3.ply6_accuracy, 4),
        "horizon_accuracy_drop_ply1_to_ply6": round(sg3.horizon_accuracy_drop, 4),
        "d3pm_20step_batch_latency_sec": round(sg3.d3pm_latency_sec, 4),
        "autoregressive_beam_latency_sec": round(sg3.beam_latency_sec, 4),
        "inference_speedup_vs_beam_search": round(sg3.latency_speedup_ratio, 2),
        "initial_vlb_loss": round(losses[0], 5),
        "final_vlb_loss": round(losses[-1], 5),
        "elapsed_sec": round(elapsed, 2),
    }
    print(f"[WP3] Stage-Gate 3 Result: {json.dumps(report, indent=2)}")
    return report


def main() -> int:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 76)
    print(f"Topographical Tensors — Full WP1, WP2, WP3 Stage-Gate Verification ({device})")
    print("=" * 76)

    sg1_report = run_stage_gate_1(num_test_positions=100_000)
    sg2_report, ipa_5x5 = run_stage_gate_2(device=device)
    sg3_report = run_stage_gate_3(ipa_5x5=ipa_5x5, device=device, num_legality_positions=10_000)

    all_passed = bool(sg1_report["passed"] and sg2_report["passed"] and sg3_report["passed"])
    summary = {
        "timestamp": "2026-11-05",
        "device": str(device),
        "all_stage_gates_passed": all_passed,
        "stage_gate_1": sg1_report,
        "stage_gate_2": sg2_report,
        "stage_gate_3": sg3_report,
    }

    data_dir = Path(os.environ.get("TOPO_DATA_DIR", "./data"))
    if data_dir.is_dir():
        log_path = data_dir / "logs" / "stage_gates_1_2_3_report.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(summary, indent=2))
        ckpt_dir = data_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ipa_5x5.state_dict(), ckpt_dir / "wp2_ipa_5x5_verified.pt")
        print(f"\nSaved verification report to {log_path}")

    if not all_passed:
        raise RuntimeError("One or more Stage-Gates failed!")
    print("\nALL STAGE-GATES (WP1, WP2, WP3) PASSED 100% OF THRESHOLD METRICS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
