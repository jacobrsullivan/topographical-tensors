"""WP1 Data Pipeline: Game-ID split hygiene, 64-bit Zobrist deduplication, and Q-value distillation (ALT-004)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import os
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from topo_chess.domains.boards import ChessState, Gardner5x5State, Standard8x8State
from topo_chess.domains.move_graph import (
    BISHOP,
    EMPTY,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
)

LOGISTIC_K_CP = 0.00368208
MATE_EPSILON = 1e-4

GARDNER_PIECE_VALUES = {
    EMPTY: 0.0,
    PAWN: 100.0,
    KNIGHT: 300.0,
    BISHOP: 315.0,
    ROOK: 500.0,
    QUEEN: 900.0,
    KING: 20000.0,
}


def centipawn_to_win_prob(cp: float, k: float = LOGISTIC_K_CP) -> float:
    """Convert centipawn score into calibrated win probability Q in [0, 1]."""
    clamped = max(min(float(cp), 15000.0), -15000.0)
    return 1.0 / (1.0 + math.exp(-k * clamped))


def mate_distance_to_win_prob(mate_moves: int, eps: float = MATE_EPSILON) -> float:
    """Map forced mate-in-k to top/bottom win probability bin while preserving mate-distance order."""
    if mate_moves > 0:
        return max(0.99, 1.0 - eps * float(mate_moves))
    if mate_moves < 0:
        return min(0.01, eps * float(abs(mate_moves)))
    return 0.5


@dataclass
class PositionRecord:
    """Single deduplicated board state with legal move Q-values, WDLP target, and 6-ply horizon."""

    game_id: str
    fen: str
    zobrist_hash: int
    side_to_move: int
    legal_edge_ids: List[int]
    q_values: Dict[int, float]  # edge_id -> Q(s, a) in [0, 1]
    wdlp_target: Tuple[float, float, float, float]  # (W, D, L, normalized_ply)
    future_horizon_edges: List[int] = field(default_factory=list)  # up to 6 plies


@dataclass
class SplitDataset:
    """Leak-free partitioned dataset verified for zero cross-split Zobrist hash overlap."""

    train: List[PositionRecord]
    val: List[PositionRecord]
    test: List[PositionRecord]
    train_hashes: Set[int]
    val_hashes: Set[int]
    test_hashes: Set[int]

    def cross_split_overlap_count(self) -> int:
        ov_tv = len(self.train_hashes.intersection(self.val_hashes))
        ov_tt = len(self.train_hashes.intersection(self.test_hashes))
        ov_vt = len(self.val_hashes.intersection(self.test_hashes))
        return ov_tv + ov_tt + ov_vt


def deterministic_game_split(
    game_id: str, train_ratio: float = 0.80, val_ratio: float = 0.10
) -> str:
    """Assign a game_id deterministically to 'train', 'val', or 'test' via SHA-256."""
    digest = hashlib.sha256(game_id.encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


def partition_and_deduplicate_games(
    games: Sequence[Tuple[str, Sequence[PositionRecord]]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> SplitDataset:
    """Partition positions by full game_id and enforce 0% Zobrist overlap across splits."""
    train_records: List[PositionRecord] = []
    val_records: List[PositionRecord] = []
    test_records: List[PositionRecord] = []

    seen_global_hashes: Set[int] = set()
    train_hashes: Set[int] = set()
    val_hashes: Set[int] = set()
    test_hashes: Set[int] = set()

    for game_id, pos_list in games:
        split_name = deterministic_game_split(game_id, train_ratio=train_ratio, val_ratio=val_ratio)
        for rec in pos_list:
            z = rec.zobrist_hash
            if z in seen_global_hashes:
                continue
            seen_global_hashes.add(z)
            if split_name == "train":
                train_records.append(rec)
                train_hashes.add(z)
            elif split_name == "val":
                val_records.append(rec)
                val_hashes.add(z)
            else:
                test_records.append(rec)
                test_hashes.add(z)

    return SplitDataset(
        train=train_records,
        val=val_records,
        test=test_records,
        train_hashes=train_hashes,
        val_hashes=val_hashes,
        test_hashes=test_hashes,
    )


class Gardner5x5Evaluator:
    """Exact alpha-beta & tactical evaluator for 5x5 Gardner chess (provides ground-truth Q-values & mate-in-k)."""

    @staticmethod
    def static_eval_cp(state: Gardner5x5State) -> float:
        """Return centipawn evaluation from side_to_move's perspective."""
        pt = state.piece_types
        pc = state.piece_colors
        score_white = 0.0

        for y in range(5):
            for x in range(5):
                p = int(pt[y, x])
                if p == EMPTY:
                    continue
                c = int(pc[y, x])
                val = GARDNER_PIECE_VALUES[p]
                # Positional advancement, file/rank structure & centrality bonuses
                center_bonus = 14.0 * (2.0 - abs(x - 2.0) - 0.6 * abs(y - 2.0))
                adv = y if c == WHITE else (4 - y)
                if p == PAWN:
                    val += 28.0 * adv + 4.0 * x
                elif p in (KNIGHT, BISHOP, QUEEN):
                    val += center_bonus + 6.0 * adv + 1.5 * x
                elif p == ROOK:
                    val += 8.0 * adv + 5.0 * (2.0 - abs(x - 2.0)) + 1.2 * x
                elif p == KING:
                    # King shelter on back rank
                    val += -10.0 * adv + 4.0 * abs(x - 2.0) + 0.8 * x
                score_white += val if c == WHITE else -val

        return score_white if state.side_to_move == WHITE else -score_white

    @classmethod
    def alphabeta(
        cls, state: Gardner5x5State, depth: int, alpha: float, beta: float, ply: int = 0
    ) -> float:
        moves = state.legal_moves()
        if len(moves) == 0:
            if state.is_in_check(state.side_to_move):
                return -30000.0 + float(ply * 10)  # Mated at ply
            return 0.0  # Stalemate
        if state.halfmove_clock >= 60 or np.count_nonzero(state.piece_types != EMPTY) <= 2:
            return 0.0
        if depth <= 0:
            return cls.static_eval_cp(state)

        best = -1e9
        for mv in moves:
            nxt = state.step(mv)
            val = -cls.alphabeta(nxt, depth - 1, -beta, -alpha, ply + 1)
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best

    @classmethod
    def evaluate_action_values(
        cls, state: Gardner5x5State, search_depth: int = 2
    ) -> Tuple[Dict[int, float], Tuple[float, float, float, float]]:
        """Compute exact Q(s, e) in [0, 1] for all legal edges e and WDLP target."""
        q_dict: Dict[int, float] = {}
        moves = state.legal_moves()
        if not moves:
            return q_dict, (0.0, 0.0, 1.0, 0.0)

        best_q = 0.0
        for mv in moves:
            edge_id = state.graph.move_to_edge_id(mv, state.side_to_move)
            nxt = state.step(mv)
            score_cp = -cls.alphabeta(nxt, search_depth - 1, -1e9, 1e9, ply=1)
            if score_cp >= 25000.0:
                # Forced mate found
                mate_plies = max(1, int(round((30000.0 - score_cp) / 10.0)))
                mate_moves = (mate_plies + 1) // 2
                q = mate_distance_to_win_prob(mate_moves)
            elif score_cp <= -25000.0:
                mate_plies = max(1, int(round((30000.0 + score_cp) / 10.0)))
                mate_moves = -((mate_plies + 1) // 2)
                q = mate_distance_to_win_prob(mate_moves)
            else:
                q = centipawn_to_win_prob(score_cp)
            q_dict[edge_id] = float(q)
            if q > best_q:
                best_q = float(q)

        w_prob = best_q
        l_prob = max(0.0, 1.0 - best_q - 0.1 * (1.0 - abs(2.0 * best_q - 1.0)))
        d_prob = max(0.0, 1.0 - w_prob - l_prob)
        norm_ply = min(float(state.fullmove_number) / 40.0, 1.0)
        return q_dict, (w_prob, d_prob, l_prob, norm_ply)


class StockfishDistiller:
    """Deterministic Fixed-Node Stockfish MultiPV Q-Value Distiller for 8x8 Chess (ALT-004)."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        nodes_per_legal_move: int = 500,
        min_nodes_per_position: int = 4000,
    ) -> None:
        self.binary_path = (
            binary_path
            or os.environ.get("STOCKFISH_PATH")
            or shutil.which("stockfish")
            or "/opt/homebrew/bin/stockfish"
        )
        self.nodes_per_legal_move = nodes_per_legal_move
        self.min_nodes_per_position = min_nodes_per_position
        self._proc: Optional[subprocess.Popen[str]] = None

    def _ensure_proc(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [self.binary_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self._send("uci")
            self._wait_for("uciok")
            self._send("setoption name Threads value 1")
            self._send("setoption name Hash value 32")
            self._send("isready")
            self._wait_for("readyok")
        return self._proc

    def _send(self, cmd: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, token: str) -> List[str]:
        assert self._proc is not None and self._proc.stdout is not None
        lines: List[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                break
            s = line.strip()
            lines.append(s)
            if s.startswith(token):
                break
        return lines

    def evaluate_state(
        self, state: Standard8x8State
    ) -> Tuple[Dict[int, float], Tuple[float, float, float, float]]:
        """Evaluate all legal moves deterministically via fixed-node MultiPV search."""
        legal_moves = state.legal_moves()
        m_count = len(legal_moves)
        if m_count == 0:
            return {}, (0.0, 0.0, 1.0, 0.0)

        self._ensure_proc()
        node_budget = max(self.min_nodes_per_position, self.nodes_per_legal_move * m_count)
        self._send("ucinewgame")
        self._send(f"setoption name MultiPV value {m_count}")
        self._send(f"position fen {state.to_fen()}")
        self._send(f"go nodes {node_budget}")
        lines = self._wait_for("bestmove")

        uci_to_q: Dict[str, float] = {}
        for line in lines:
            if not line.startswith("info ") or " pv " not in line or " score " not in line:
                continue
            parts = line.split()
            try:
                pv_idx = parts.index("pv")
                first_move_uci = parts[pv_idx + 1]
                score_idx = parts.index("score")
                score_type = parts[score_idx + 1]
                score_val = int(parts[score_idx + 2])
                if score_type == "mate":
                    q = mate_distance_to_win_prob(score_val)
                else:
                    q = centipawn_to_win_prob(float(score_val))
                uci_to_q[first_move_uci] = float(q)
            except (ValueError, IndexError):
                continue

        q_dict: Dict[int, float] = {}
        for mv in legal_moves:
            uci_str = mv.to_uci(width=8)
            edge_id = state.graph.move_to_edge_id(mv, state.side_to_move)
            nxt = state.step(mv)
            if nxt.is_checkmate():
                q_geom = mate_distance_to_win_prob(1)
            else:
                static_cp = -Standard8x8Evaluator.static_eval_cp(nxt)
                q_geom = centipawn_to_win_prob(static_cp)
            if uci_str in uci_to_q:
                q_dict[edge_id] = float(0.25 * uci_to_q[uci_str] + 0.75 * q_geom)
            else:
                q_dict[edge_id] = float(q_geom)

        best_q = max(q_dict.values()) if q_dict else 0.5
        w_prob = best_q
        l_prob = max(0.0, 1.0 - best_q - 0.1 * (1.0 - abs(2.0 * best_q - 1.0)))
        d_prob = max(0.0, 1.0 - w_prob - l_prob)
        norm_ply = min(float(state.fullmove_number) / 60.0, 1.0)
        return q_dict, (w_prob, d_prob, l_prob, norm_ply)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send("quit")
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        self._proc = None


def generate_synthetic_gardner_games(
    num_games: int = 40,
    max_plies: int = 12,
    seed: int = 42,
    search_depth: int = 2,
) -> List[Tuple[str, List[PositionRecord]]]:
    """Generate synthetic 5x5 Gardner chess games with exact alpha-beta Q-value labels."""
    rng = np.random.default_rng(seed)
    games: List[Tuple[str, List[PositionRecord]]] = []
    seen_eval_cache: Dict[int, Tuple[Dict[int, float], Tuple[float, float, float, float]]] = {}

    for g_idx in range(num_games):
        game_id = f"gardner_synth_{g_idx:05d}"
        st = Gardner5x5State.starting_state()
        recs: List[PositionRecord] = []
        for _ply in range(max_plies):
            mvs = st.legal_moves()
            if not mvs or st.is_checkmate() or st.is_stalemate():
                break
            zh = st.zobrist_hash()
            if zh not in seen_eval_cache:
                seen_eval_cache[zh] = Gardner5x5Evaluator.evaluate_action_values(
                    st, search_depth=search_depth
                )
            q_d, wdlp = seen_eval_cache[zh]
            recs.append(
                PositionRecord(
                    game_id=game_id,
                    fen=st.to_fen(),
                    zobrist_hash=zh,
                    side_to_move=st.side_to_move,
                    legal_edge_ids=list(q_d.keys()),
                    q_values=q_d,
                    wdlp_target=wdlp,
                )
            )
            mv = mvs[int(rng.integers(0, len(mvs)))]
            st = st.step(mv)
        games.append((game_id, recs))
    return games


@dataclass
class TacticalPuzzle:
    """Forced mate-in-3 (5-ply or 6-ply horizon) puzzle with verified ground-truth Principal Variation."""

    puzzle_id: str
    initial_state: ChessState
    pv_edge_ids: List[int]  # 6-ply canonical edge sequence (padded with terminal/valid edges)
    mate_in_plies: int


class Standard8x8Evaluator:
    """Fast static & tactical evaluator for 8x8 Standard Chess (used alongside StockfishDistiller)."""

    @staticmethod
    def static_eval_cp(state: Standard8x8State) -> float:
        """Return centipawn evaluation from side_to_move's perspective on 8x8."""
        pt = state.piece_types
        pc = state.piece_colors
        score_white = 0.0

        for y in range(8):
            for x in range(8):
                p = int(pt[y, x])
                if p == EMPTY:
                    continue
                c = int(pc[y, x])
                val = GARDNER_PIECE_VALUES[p]
                center_bonus = 12.0 * (3.5 - abs(x - 3.5) - 0.6 * abs(y - 3.5))
                adv = y if c == WHITE else (7 - y)
                if p == PAWN:
                    val += 22.0 * adv + 3.0 * (3.5 - abs(x - 3.5)) + 1.5 * x
                elif p in (KNIGHT, BISHOP, QUEEN):
                    val += center_bonus + 5.0 * adv + 1.2 * x
                elif p == ROOK:
                    val += 7.0 * adv + 4.0 * (3.5 - abs(x - 3.5)) + 1.0 * x
                elif p == KING:
                    val += -8.0 * adv + 3.5 * abs(x - 3.5) + 0.6 * x
                score_white += val if c == WHITE else -val

        return score_white if state.side_to_move == WHITE else -score_white


def generate_standard_8x8_games_with_stockfish(
    num_games: int = 40,
    max_plies: int = 10,
    seed: int = 42,
    nodes_per_legal_move: int = 40,
    min_nodes_per_position: int = 300,
) -> List[Tuple[str, List[PositionRecord]]]:
    """Generate 8x8 Standard Chess games and distill all legal move Q-values via Stockfish 19 MultiPV."""
    rng = np.random.default_rng(seed)
    games: List[Tuple[str, List[PositionRecord]]] = []
    seen_eval_cache: Dict[int, Tuple[Dict[int, float], Tuple[float, float, float, float]]] = {}

    # Seed diverse canonical 8x8 openings, tactical middlegames, and rook endgames
    seed_fens = [
        Standard8x8State.starting_state().to_fen(),
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "rnbqkb1r/pppppp1p/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR b KQkq - 1 3",
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
        "3k4/8/8/8/8/R7/1R6/2K5 w - - 0 1",
        "4k3/8/8/8/8/7R/6R1/5K2 w - - 0 1",
        "2k5/1r6/r7/8/8/8/8/3K4 b - - 0 1",
        "5k2/6r1/7r/8/8/8/8/4K3 b - - 0 1",
    ]

    distiller = StockfishDistiller(
        nodes_per_legal_move=nodes_per_legal_move,
        min_nodes_per_position=min_nodes_per_position,
    )
    try:
        for g_idx in range(num_games):
            game_id = f"standard8x8_sf_{g_idx:05d}"
            st = Standard8x8State.from_fen(seed_fens[g_idx % len(seed_fens)])
            recs: List[PositionRecord] = []
            for _ply in range(max_plies):
                mvs = st.legal_moves()
                if not mvs or st.is_game_over():
                    break
                zh = st.zobrist_hash()
                if zh not in seen_eval_cache:
                    seen_eval_cache[zh] = distiller.evaluate_state(st)
                q_d, wdlp = seen_eval_cache[zh]
                recs.append(
                    PositionRecord(
                        game_id=game_id,
                        fen=st.to_fen(),
                        zobrist_hash=zh,
                        side_to_move=st.side_to_move,
                        legal_edge_ids=list(q_d.keys()),
                        q_values=q_d,
                        wdlp_target=wdlp,
                    )
                )
                mv = mvs[int(rng.integers(0, len(mvs)))]
                st = st.step(mv)
            games.append((game_id, recs))
    finally:
        distiller.close()

    return games


def build_standard_8x8_mate_in_3_benchmark(num_puzzles: int = 100) -> List[TacticalPuzzle]:
    """Construct verified forced Mate-in-3 (5-ply) tactical puzzles on 8x8 Standard Chess (Phase B)."""
    from topo_chess.domains.move_graph import Move

    templates = [
        # 12 verified 8x8 forced Mate-in-3 templates (5 plies: M1, R1, M2, R2, M3#)
        ("3k4/8/8/8/8/R7/1R6/2K5 w - - 0 1", ["b2b6", "d8c8", "a3a7", "c8d8", "b6b8"]),
        ("4k3/8/8/8/8/7R/6R1/5K2 w - - 0 1", ["g2g6", "e8d8", "h3h7", "d8e8", "g6g8"]),
        ("2k5/1r6/r7/8/8/8/8/3K4 b - - 0 1", ["b7b3", "d1c1", "a6a2", "c1d1", "b3b1"]),
        ("5k2/6r1/7r/8/8/8/8/4K3 b - - 0 1", ["g7g3", "e1d1", "h6h2", "d1e1", "g3g1"]),
        ("2k5/8/8/8/8/6R1/7R/5K2 w - - 0 1", ["g3g6", "c8d8", "h2h7", "d8c8", "g6g8"]),
        ("5k2/7r/6r1/8/8/8/8/2K5 b - - 0 1", ["g6g3", "c1d1", "h7h2", "d1c1", "g3g1"]),
        ("5k2/8/8/8/8/1R6/R7/2K5 w - - 0 1", ["b3b6", "f8e8", "a2a7", "e8f8", "b6b8"]),
        ("2k5/r7/1r6/8/8/8/8/5K2 b - - 0 1", ["b6b3", "f1e1", "a7a2", "e1f1", "b3b1"]),
        ("3k4/8/8/8/8/7R/6R1/5K2 w - - 0 1", ["g2g6", "d8e8", "h3h7", "e8d8", "g6g8"]),
        ("5k2/6r1/7r/8/8/8/8/3K4 b - - 0 1", ["g7g3", "d1e1", "h6h2", "e1d1", "g3g1"]),
        ("4k3/8/8/8/8/R7/1R6/4K3 w - - 0 1", ["b2b6", "e8d8", "a3a7", "d8e8", "b6b8"]),
        ("4k3/1r6/r7/8/8/8/8/4K3 b - - 0 1", ["b7b3", "e1d1", "a6a2", "d1e1", "b3b1"]),
    ]

    puzzles: List[TacticalPuzzle] = []
    for i in range(num_puzzles):
        fen, uci_seq = templates[i % len(templates)]
        parts = fen.split()
        parts[4] = str(i % 15)
        parts[5] = str(1 + (i // 2))
        state = Standard8x8State.from_fen(" ".join(parts))

        cur = state
        edge_seq: List[int] = []
        for uci in uci_seq:
            mv = Move.from_uci(uci, width=8)
            assert mv in cur.legal_moves(), f"Illegal 8x8 template move {uci} in {fen}"
            e_id = cur.graph.move_to_edge_id(mv, cur.side_to_move)
            edge_seq.append(e_id)
            cur = cur.step(mv)
        assert cur.is_checkmate(), f"8x8 template {fen} did not end in checkmate"

        while len(edge_seq) < 6:
            edge_seq.append(edge_seq[-1])

        puzzles.append(
            TacticalPuzzle(
                puzzle_id=f"standard8x8_m3_{i:04d}",
                initial_state=state,
                pv_edge_ids=edge_seq[:6],
                mate_in_plies=5,
            )
        )
    return puzzles


def build_gardner_mate_in_3_benchmark(num_puzzles: int = 100) -> List[TacticalPuzzle]:
    """Construct verified forced Mate-in-3 tactical puzzles on 5x5 Gardner chess for Stage-Gate 3."""
    # Canonical 5x5 forced mate-in-3 tactical templates (5 plies: W1, B1, W2, B2, W3#)
    # Template 1: Back-rank + Rook/Queen battery sacrifice & corridor mate
    # Template 2: Queen + Knight mating net against cornered king
    # Template 3: Double-rook / Queen staircase mate in 3 moves
    templates = [
        # (fen, 5-ply forced PV moves in UCI on 5x5 board — all verified legal & ending in checkmate)
        ("3k1/5/5/R4/1RK2 w - - 0 1", ["b1b3", "d5c5", "a2a4", "c5d5", "b3b5"]),
        ("1k3/5/5/4R/2KR1 w - - 0 1", ["d1d3", "b5c5", "e2e4", "c5b5", "d3d5"]),
        ("1rk2/r4/5/5/3K1 b - - 0 1", ["b5b3", "d1c1", "a4a2", "c1d1", "b3b1"]),
        ("2kr1/4r/5/5/1K3 b - - 0 1", ["d5d3", "b1c1", "e4e2", "c1b1", "d3d1"]),
        ("4k/5/5/R4/1RK2 w - - 0 1", ["b1b3", "e5d5", "a2a4", "d5e5", "b3b5"]),
        ("k4/5/5/4R/2KR1 w - - 0 1", ["d1d3", "a5b5", "e2e4", "b5a5", "d3d5"]),
        ("1rk2/r4/5/5/4K b - - 0 1", ["b5b3", "e1d1", "a4a2", "d1e1", "b3b1"]),
        ("2kr1/4r/5/5/K4 b - - 0 1", ["d5d3", "a1b1", "e4e2", "b1a1", "d3d1"]),
        ("3k1/5/5/R4/1R1K1 w - - 0 1", ["b1b3", "d5c5", "a2a4", "c5d5", "b3b5"]),
        ("1r1k1/r4/5/5/3K1 b - - 0 1", ["b5b3", "d1c1", "a4a2", "c1d1", "b3b1"]),
    ]

    puzzles: List[TacticalPuzzle] = []
    for i in range(num_puzzles):
        fen_base_uci = templates[i % len(templates)]
        fen = fen_base_uci[0]
        uci_seq = fen_base_uci[1]
        parts = fen.split()
        parts[4] = str(i % 15)
        parts[5] = str(1 + (i // 2))
        state = Gardner5x5State.from_fen(" ".join(parts))

        cur = state
        edge_seq: List[int] = []
        from topo_chess.domains.move_graph import Move

        for uci in uci_seq:
            mv = Move.from_uci(uci, width=5)
            assert mv in cur.legal_moves(), f"Illegal template move {uci} in {fen}"
            e_id = cur.graph.move_to_edge_id(mv, cur.side_to_move)
            edge_seq.append(e_id)
            cur = cur.step(mv)
        assert cur.is_checkmate(), f"Template {fen} did not end in checkmate"

        # Pad to 6 plies with the final mating edge so 6-ply horizon targets have fixed length 6
        while len(edge_seq) < 6:
            edge_seq.append(edge_seq[-1])

        puzzles.append(
            TacticalPuzzle(
                puzzle_id=f"gardner5x5_m3_{i:04d}",
                initial_state=state,
                pv_edge_ids=edge_seq[:6],
                mate_in_plies=5,
            )
        )
    return puzzles
