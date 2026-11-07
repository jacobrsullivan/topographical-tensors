"""Stage-Gate 1 Verification Suite: Data Hygiene, Move Graph Exactness, and Reversible Topographical Encoding."""

from __future__ import annotations

from pathlib import Path
import random
from typing import List, Tuple
import chess
import yaml

from topo_chess.data.pipeline import (
    Gardner5x5Evaluator,
    PositionRecord,
    StockfishDistiller,
    partition_and_deduplicate_games,
)
from topo_chess.domains.boards import Gardner5x5State, Standard8x8State
from topo_chess.domains.move_graph import GARDNER_5X5_GRAPH, STANDARD_8X8_GRAPH
from topo_chess.encoding.topographical_encoder import TopographicalDecoder, TopographicalEncoder


def test_sg1_move_graph_exact_edge_counts() -> None:
    """SG1-B: Verify 5x5 Gardner has 455 valid edges and 8x8 Standard has 1,858 valid edges."""
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "wp1_data_pipeline.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())

    assert GARDNER_5X5_GRAPH.theoretical_actions == 1225
    assert GARDNER_5X5_GRAPH.num_edges == 455
    assert STANDARD_8X8_GRAPH.num_edges == 1858
    assert cfg["domains"]["gardner_5x5"]["valid_move_edges"] == GARDNER_5X5_GRAPH.num_edges
    assert cfg["domains"]["standard_8x8"]["valid_move_edges"] == STANDARD_8X8_GRAPH.num_edges


def test_sg1_split_hygiene_zero_zobrist_overlap() -> None:
    """SG1-A: Verify partitioning by game_id and 64-bit Zobrist deduplication yields 0% cross-split overlap."""
    rng = random.Random(42)
    games: List[Tuple[str, List[PositionRecord]]] = []

    for g_idx in range(60):
        game_id = f"game_{g_idx:04d}"
        st = Gardner5x5State.starting_state()
        records: List[PositionRecord] = []
        for _ply in range(12):
            moves = st.legal_moves()
            if not moves:
                break
            edges = st.legal_edge_indices()
            q_map = {e: 0.5 for e in edges}
            records.append(
                PositionRecord(
                    game_id=game_id,
                    fen=st.to_fen(),
                    zobrist_hash=st.zobrist_hash(),
                    side_to_move=st.side_to_move,
                    legal_edge_ids=edges,
                    q_values=q_map,
                    wdlp_target=(0.4, 0.2, 0.4, 0.2),
                )
            )
            st = st.step(rng.choice(moves))
        games.append((game_id, records))

    split_ds = partition_and_deduplicate_games(games, train_ratio=0.80, val_ratio=0.10)
    assert split_ds.cross_split_overlap_count() == 0
    assert len(split_ds.train) > 0 and len(split_ds.val) > 0 and len(split_ds.test) > 0


def test_sg1_reversible_topographical_encoding_5x5_and_8x8() -> None:
    """SG1-C: Verify Decode(Encode(s)) reconstructs 100% of legal moves on both 5x5 and 8x8 positions."""
    rng = random.Random(1337)

    # 1. Test 5x5 Gardner random rollouts (including promotions and checks)
    enc5 = TopographicalEncoder(height=5, width=5)
    dec5 = TopographicalDecoder(height=5, width=5)
    st5 = Gardner5x5State.starting_state()
    for _ in range(500):
        tensor5 = enc5.encode(st5)
        assert tensor5.shape == (40, 5, 5)
        recon5 = dec5.decode(tensor5)
        assert set(recon5.legal_edge_indices()) == set(st5.legal_edge_indices())
        assert recon5.zobrist_hash() == st5.zobrist_hash()
        moves = st5.legal_moves()
        if not moves or st5.is_game_over():
            st5 = Gardner5x5State.starting_state()
        else:
            st5 = st5.step(rng.choice(moves))

    # 2. Test 8x8 Standard random rollouts (including castling rights, en passant, promotions, X-rays)
    enc8 = TopographicalEncoder(height=8, width=8)
    dec8 = TopographicalDecoder(height=8, width=8)
    special_fens = [
        chess.STARTING_FEN,
        # Position with legal en passant capture and castling rights
        "r1bqkbnr/ppp1pppp/2n5/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
        # Position with promotions & underpromotions on 7th rank
        "1r2k3/P1P5/8/8/8/8/6p1/4K2R w K - 0 1",
        # Position with pinned piece along Rook/Bishop X-ray
        "4k3/8/4r3/8/8/4N3/8/4K3 w - - 4 10",
    ]
    for fen in special_fens:
        st8 = Standard8x8State.from_fen(fen)
        t8 = enc8.encode(st8)
        assert t8.shape == (44, 8, 8)
        rec8 = dec8.decode(t8)
        assert set(rec8.legal_edge_indices()) == set(st8.legal_edge_indices())
        assert rec8.zobrist_hash() == st8.zobrist_hash()

    st8 = Standard8x8State.starting_state()
    for _ in range(500):
        t8 = enc8.encode(st8)
        rec8 = dec8.decode(t8)
        assert set(rec8.legal_edge_indices()) == set(st8.legal_edge_indices())
        assert rec8.zobrist_hash() == st8.zobrist_hash()
        moves = st8.legal_moves()
        if not moves or st8.is_game_over():
            st8 = Standard8x8State.starting_state()
        else:
            st8 = st8.step(rng.choice(moves))


def test_sg1_deterministic_q_value_distillation() -> None:
    """ALT-004: Verify deterministic fixed-node Q-value evaluation for 5x5 and 8x8."""
    st5 = Gardner5x5State.starting_state()
    q1, wdlp1 = Gardner5x5Evaluator.evaluate_action_values(st5, search_depth=2)
    q2, wdlp2 = Gardner5x5Evaluator.evaluate_action_values(st5, search_depth=2)
    assert q1 == q2 and wdlp1 == wdlp2
    assert len(q1) == len(st5.legal_moves())

    distiller = StockfishDistiller(nodes_per_legal_move=200, min_nodes_per_position=2000)
    if Path(distiller.binary_path).is_file():
        st8 = Standard8x8State.starting_state()
        sq1, _ = distiller.evaluate_state(st8)
        sq2, _ = distiller.evaluate_state(st8)
        distiller.close()
        assert len(sq1) == 20
        for k in sq1:
            assert abs(sq1[k] - sq2[k]) < 1e-6
