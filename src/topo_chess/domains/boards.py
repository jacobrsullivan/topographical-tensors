"""Board state implementations for 5x5 Gardner Chess and 8x8 Standard Chess with 64-bit Zobrist hashing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import chess
import numpy as np

from topo_chess.domains.move_graph import (
    ALL_PROMOTION_PIECES,
    BISHOP,
    BLACK,
    EMPTY,
    GARDNER_5X5_GRAPH,
    KING,
    KNIGHT,
    Move,
    MoveGraph,
    PAWN,
    QUEEN,
    ROOK,
    STANDARD_8X8_GRAPH,
    WHITE,
)

PIECE_CHARS = {
    (PAWN, WHITE): "P",
    (KNIGHT, WHITE): "N",
    (BISHOP, WHITE): "B",
    (ROOK, WHITE): "R",
    (QUEEN, WHITE): "Q",
    (KING, WHITE): "K",
    (PAWN, BLACK): "p",
    (KNIGHT, BLACK): "n",
    (BISHOP, BLACK): "b",
    (ROOK, BLACK): "r",
    (QUEEN, BLACK): "q",
    (KING, BLACK): "k",
}
CHAR_TO_PIECE: Dict[str, Tuple[int, int]] = {v: k for k, v in PIECE_CHARS.items()}


class ZobristHasher:
    """Deterministic 64-bit Zobrist hash generator using SplitMix64 PRNG."""

    def __init__(self, max_squares: int = 64, seed: int = 0x9E3779B97F4A7C15) -> None:
        self.max_squares = max_squares
        state = seed & 0xFFFFFFFFFFFFFFFF

        def next_u64() -> int:
            nonlocal state
            state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
            return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

        # piece_keys[color][piece_type][sq]
        self.piece_keys = np.zeros((2, 7, max_squares), dtype=np.uint64)
        for c in range(2):
            for p in range(1, 7):
                for sq in range(max_squares):
                    self.piece_keys[c, p, sq] = np.uint64(next_u64())

        self.side_key = np.uint64(next_u64())
        self.castling_keys = np.array([next_u64() for _ in range(16)], dtype=np.uint64)
        self.ep_keys = np.array([next_u64() for _ in range(max_squares + 1)], dtype=np.uint64)

    def compute(
        self,
        piece_types: np.ndarray,
        piece_colors: np.ndarray,
        side_to_move: int,
        castling_mask: int = 0,
        ep_square: Optional[int] = None,
    ) -> int:
        h = np.uint64(0)
        flat_p = piece_types.reshape(-1)
        flat_c = piece_colors.reshape(-1)
        for sq in range(flat_p.shape[0]):
            p = int(flat_p[sq])
            if p != EMPTY:
                c = int(flat_c[sq])
                h ^= self.piece_keys[c, p, sq]
        if side_to_move == BLACK:
            h ^= self.side_key
        h ^= self.castling_keys[castling_mask & 0xF]
        if ep_square is not None and 0 <= ep_square < self.max_squares:
            h ^= self.ep_keys[ep_square]
        return int(h)


GLOBAL_ZOBRIST = ZobristHasher(max_squares=64)


class Gardner5x5State:
    """Complete, exact 5x5 Gardner Minichess board state and legal move engine."""

    STARTING_FEN = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1"

    def __init__(
        self,
        piece_types: np.ndarray,
        piece_colors: np.ndarray,
        side_to_move: int = WHITE,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
    ) -> None:
        # Shape (5, 5), indexed [rank y, file x] where y=0 is rank 1 (White back rank)
        self.piece_types = piece_types.astype(np.int8, copy=True)
        self.piece_colors = piece_colors.astype(np.int8, copy=True)
        self.side_to_move = int(side_to_move)
        self.halfmove_clock = int(halfmove_clock)
        self.fullmove_number = int(fullmove_number)
        self._legal_moves_cache: Optional[List[Move]] = None

    @property
    def height(self) -> int:
        return 5

    @property
    def width(self) -> int:
        return 5

    @property
    def graph(self) -> MoveGraph:
        return GARDNER_5X5_GRAPH

    @property
    def castling_mask(self) -> int:
        return 0

    @property
    def ep_square(self) -> Optional[int]:
        return None

    @classmethod
    def starting_state(cls) -> "Gardner5x5State":
        return cls.from_fen(cls.STARTING_FEN)

    @classmethod
    def initial(cls) -> "Gardner5x5State":
        return cls.starting_state()

    def clone(self) -> "Gardner5x5State":
        return Gardner5x5State(
            piece_types=self.piece_types,
            piece_colors=self.piece_colors,
            side_to_move=self.side_to_move,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )

    @classmethod
    def from_fen(cls, fen: str) -> "Gardner5x5State":
        parts = fen.strip().split()
        board_str = parts[0]
        stm = WHITE if (len(parts) < 2 or parts[1] == "w") else BLACK
        hm = int(parts[4]) if len(parts) >= 5 else 0
        fm = int(parts[5]) if len(parts) >= 6 else 1

        pt = np.zeros((5, 5), dtype=np.int8)
        pc = np.zeros((5, 5), dtype=np.int8)
        ranks = board_str.split("/")
        if len(ranks) != 5:
            raise ValueError(f"Expected 5 ranks in 5x5 FEN, got {len(ranks)}: {fen}")

        for r_idx, rank_str in enumerate(ranks):
            y = 4 - r_idx
            x = 0
            for ch in rank_str:
                if ch.isdigit():
                    x += int(ch)
                else:
                    p_type, p_color = CHAR_TO_PIECE[ch]
                    pt[y, x] = p_type
                    pc[y, x] = p_color
                    x += 1
        return cls(pt, pc, side_to_move=stm, halfmove_clock=hm, fullmove_number=fm)

    def to_fen(self) -> str:
        rank_strs = []
        for y in range(4, -1, -1):
            empty_cnt = 0
            s = ""
            for x in range(5):
                p = int(self.piece_types[y, x])
                if p == EMPTY:
                    empty_cnt += 1
                else:
                    if empty_cnt > 0:
                        s += str(empty_cnt)
                        empty_cnt = 0
                    c = int(self.piece_colors[y, x])
                    s += PIECE_CHARS[(p, c)]
            if empty_cnt > 0:
                s += str(empty_cnt)
            rank_strs.append(s)
        stm_ch = "w" if self.side_to_move == WHITE else "b"
        return f"{'/'.join(rank_strs)} {stm_ch} - - {self.halfmove_clock} {self.fullmove_number}"

    def zobrist_hash(self) -> int:
        return GLOBAL_ZOBRIST.compute(
            self.piece_types, self.piece_colors, self.side_to_move, castling_mask=0, ep_square=None
        )

    def is_square_attacked_by(self, sq: int, attacker_color: int) -> bool:
        """Return True if square sq (0..24) is attacked by any piece of attacker_color."""
        tx, ty = sq % 5, sq // 5

        # 1. Pawn attacks
        pawn_dy = -1 if attacker_color == WHITE else 1  # square where attacking pawn sits
        py = ty + pawn_dy
        if 0 <= py < 5:
            for dx in (-1, 1):
                px = tx + dx
                if 0 <= px < 5:
                    if (
                        self.piece_types[py, px] == PAWN
                        and self.piece_colors[py, px] == attacker_color
                    ):
                        return True

        # 2. Knight attacks
        for dx, dy in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
            nx, ny = tx + dx, ty + dy
            if 0 <= nx < 5 and 0 <= ny < 5:
                if (
                    self.piece_types[ny, nx] == KNIGHT
                    and self.piece_colors[ny, nx] == attacker_color
                ):
                    return True

        # 3. King attacks
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                kx, ky = tx + dx, ty + dy
                if 0 <= kx < 5 and 0 <= ky < 5:
                    if (
                        self.piece_types[ky, kx] == KING
                        and self.piece_colors[ky, kx] == attacker_color
                    ):
                        return True

        # 4. Orthogonal rays (Rook / Queen)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            for step in range(1, 5):
                rx, ry = tx + dx * step, ty + dy * step
                if not (0 <= rx < 5 and 0 <= ry < 5):
                    break
                p = int(self.piece_types[ry, rx])
                if p != EMPTY:
                    if self.piece_colors[ry, rx] == attacker_color and p in (ROOK, QUEEN):
                        return True
                    break

        # 5. Diagonal rays (Bishop / Queen)
        for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            for step in range(1, 5):
                bx, by = tx + dx * step, ty + dy * step
                if not (0 <= bx < 5 and 0 <= by < 5):
                    break
                p = int(self.piece_types[by, bx])
                if p != EMPTY:
                    if self.piece_colors[by, bx] == attacker_color and p in (BISHOP, QUEEN):
                        return True
                    break

        return False

    def find_king(self, color: int) -> Optional[int]:
        for y in range(5):
            for x in range(5):
                if self.piece_types[y, x] == KING and self.piece_colors[y, x] == color:
                    return y * 5 + x
        return None

    def is_in_check(self, color: Optional[int] = None) -> bool:
        c = self.side_to_move if color is None else color
        k_sq = self.find_king(c)
        if k_sq is None:
            return True
        return self.is_square_attacked_by(k_sq, 1 - c)

    def _generate_pseudo_legal_moves(self) -> List[Move]:
        moves: List[Move] = []
        stm = self.side_to_move
        opp = 1 - stm
        pawn_dir = 1 if stm == WHITE else -1
        promo_rank = 4 if stm == WHITE else 0

        for fy in range(5):
            for fx in range(5):
                p = int(self.piece_types[fy, fx])
                if p == EMPTY or self.piece_colors[fy, fx] != stm:
                    continue
                from_sq = fy * 5 + fx

                if p == PAWN:
                    ty = fy + pawn_dir
                    if 0 <= ty < 5:
                        # Single forward push
                        if self.piece_types[ty, fx] == EMPTY:
                            to_sq = ty * 5 + fx
                            if ty == promo_rank:
                                for promo in ALL_PROMOTION_PIECES:
                                    moves.append(Move(from_sq, to_sq, promo))
                            else:
                                moves.append(Move(from_sq, to_sq, None))
                        # Captures
                        for dx in (-1, 1):
                            tx = fx + dx
                            if 0 <= tx < 5:
                                if (
                                    self.piece_types[ty, tx] != EMPTY
                                    and self.piece_colors[ty, tx] == opp
                                ):
                                    to_sq = ty * 5 + tx
                                    if ty == promo_rank:
                                        for promo in ALL_PROMOTION_PIECES:
                                            moves.append(Move(from_sq, to_sq, promo))
                                    else:
                                        moves.append(Move(from_sq, to_sq, None))

                elif p == KNIGHT:
                    for dx, dy in (
                        (1, 2),
                        (2, 1),
                        (2, -1),
                        (1, -2),
                        (-1, -2),
                        (-2, -1),
                        (-2, 1),
                        (-1, 2),
                    ):
                        tx, ty = fx + dx, fy + dy
                        if 0 <= tx < 5 and 0 <= ty < 5:
                            if (
                                self.piece_types[ty, tx] == EMPTY
                                or self.piece_colors[ty, tx] == opp
                            ):
                                moves.append(Move(from_sq, ty * 5 + tx, None))

                elif p == KING:
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            tx, ty = fx + dx, fy + dy
                            if 0 <= tx < 5 and 0 <= ty < 5:
                                if (
                                    self.piece_types[ty, tx] == EMPTY
                                    or self.piece_colors[ty, tx] == opp
                                ):
                                    moves.append(Move(from_sq, ty * 5 + tx, None))

                else:
                    dirs = []
                    if p in (ROOK, QUEEN):
                        dirs.extend([(1, 0), (-1, 0), (0, 1), (0, -1)])
                    if p in (BISHOP, QUEEN):
                        dirs.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
                    for dx, dy in dirs:
                        for step in range(1, 5):
                            tx, ty = fx + dx * step, fy + dy * step
                            if not (0 <= tx < 5 and 0 <= ty < 5):
                                break
                            target_p = int(self.piece_types[ty, tx])
                            if target_p == EMPTY:
                                moves.append(Move(from_sq, ty * 5 + tx, None))
                            else:
                                if self.piece_colors[ty, tx] == opp:
                                    moves.append(Move(from_sq, ty * 5 + tx, None))
                                break
        return moves

    def step_unchecked(self, move: Move) -> "Gardner5x5State":
        pt = self.piece_types.copy()
        pc = self.piece_colors.copy()
        fx, fy = move.from_sq % 5, move.from_sq // 5
        tx, ty = move.to_sq % 5, move.to_sq // 5

        moving_p = int(pt[fy, fx])
        captured_p = int(pt[ty, tx])
        pt[fy, fx] = EMPTY
        pc[fy, fx] = 0

        pt[ty, tx] = int(move.promotion) if move.promotion is not None else moving_p
        pc[ty, tx] = self.side_to_move

        new_hm = 0 if (moving_p == PAWN or captured_p != EMPTY) else (self.halfmove_clock + 1)
        new_fm = self.fullmove_number + (1 if self.side_to_move == BLACK else 0)
        return Gardner5x5State(
            pt, pc, side_to_move=1 - self.side_to_move, halfmove_clock=new_hm, fullmove_number=new_fm
        )

    def legal_moves(self) -> List[Move]:
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache
        legal: List[Move] = []
        stm = self.side_to_move
        for mv in self._generate_pseudo_legal_moves():
            nxt = self.step_unchecked(mv)
            if not nxt.is_in_check(stm):
                legal.append(mv)
        self._legal_moves_cache = legal
        return legal

    def step(self, move: Move) -> "Gardner5x5State":
        return self.step_unchecked(move)

    def step_by_edge_id(self, edge_id: int) -> "Gardner5x5State":
        edge = self.graph.edges[edge_id]
        from_sq = (
            edge.from_sq if self.side_to_move == WHITE else self.graph.flip_rank_sq(edge.from_sq)
        )
        fx, fy = from_sq % 5, from_sq // 5
        is_pawn = int(self.piece_types[fy, fx]) == PAWN
        mv = self.graph.edge_id_to_move(edge_id, self.side_to_move, moving_piece_is_pawn=is_pawn)
        return self.step(mv)

    def legal_edge_indices(self) -> List[int]:
        return [self.graph.move_to_edge_id(mv, self.side_to_move) for mv in self.legal_moves()]

    def legal_edge_mask(self) -> np.ndarray:
        mask = np.zeros(self.graph.num_edges, dtype=bool)
        for idx in self.legal_edge_indices():
            mask[idx] = True
        return mask

    def is_checkmate(self) -> bool:
        return len(self.legal_moves()) == 0 and self.is_in_check(self.side_to_move)

    def is_stalemate(self) -> bool:
        return len(self.legal_moves()) == 0 and not self.is_in_check(self.side_to_move)

    def is_game_over(self) -> bool:
        if len(self.legal_moves()) == 0 or self.halfmove_clock >= 60:
            return True
        # Insufficient material (only two kings left)
        if np.count_nonzero(self.piece_types != EMPTY) <= 2:
            return True
        return False

    def mirror_vertical_swap_colors(self) -> "Gardner5x5State":
        """Reflect board across horizontal midline (y -> 4 - y) and swap White <-> Black."""
        pt = np.flip(self.piece_types, axis=0).copy()
        pc = np.zeros_like(self.piece_colors)
        flipped_c = np.flip(self.piece_colors, axis=0)
        occ = pt != EMPTY
        pc[occ] = 1 - flipped_c[occ]
        return Gardner5x5State(
            pt,
            pc,
            side_to_move=1 - self.side_to_move,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )


class Standard8x8State:
    """8x8 Standard Chess state wrapper around python-chess with unified MoveGraph & Zobrist interface."""

    def __init__(self, board: Optional[chess.Board] = None) -> None:
        self.board = board.copy(stack=False) if board is not None else chess.Board()
        self._sync_arrays()

    def _sync_arrays(self) -> None:
        self.piece_types = np.zeros((8, 8), dtype=np.int8)
        self.piece_colors = np.zeros((8, 8), dtype=np.int8)
        for sq, piece in self.board.piece_map().items():
            x, y = sq % 8, sq // 8
            self.piece_types[y, x] = int(piece.piece_type)
            self.piece_colors[y, x] = WHITE if piece.color == chess.WHITE else BLACK

    @property
    def height(self) -> int:
        return 8

    @property
    def width(self) -> int:
        return 8

    @property
    def graph(self) -> MoveGraph:
        return STANDARD_8X8_GRAPH

    @property
    def side_to_move(self) -> int:
        return WHITE if self.board.turn == chess.WHITE else BLACK

    @property
    def halfmove_clock(self) -> int:
        return int(self.board.halfmove_clock)

    @property
    def fullmove_number(self) -> int:
        return int(self.board.fullmove_number)

    @property
    def castling_mask(self) -> int:
        mask = 0
        if self.board.has_kingside_castling_rights(chess.WHITE):
            mask |= 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            mask |= 2
        if self.board.has_kingside_castling_rights(chess.BLACK):
            mask |= 4
        if self.board.has_queenside_castling_rights(chess.BLACK):
            mask |= 8
        return mask

    @property
    def ep_square(self) -> Optional[int]:
        if self.board.ep_square is not None and self.board.has_legal_en_passant():
            return int(self.board.ep_square)
        return None

    @classmethod
    def starting_state(cls) -> "Standard8x8State":
        return cls(chess.Board())

    @classmethod
    def initial(cls) -> "Standard8x8State":
        return cls.starting_state()

    def clone(self) -> "Standard8x8State":
        return Standard8x8State(self.board.copy(stack=False))

    @classmethod
    def from_fen(cls, fen: str) -> "Standard8x8State":
        return cls(chess.Board(fen))

    def to_fen(self) -> str:
        return self.board.fen()

    def zobrist_hash(self) -> int:
        return GLOBAL_ZOBRIST.compute(
            self.piece_types,
            self.piece_colors,
            self.side_to_move,
            castling_mask=self.castling_mask,
            ep_square=self.ep_square,
        )

    def is_square_attacked_by(self, sq: int, attacker_color: int) -> bool:
        c = chess.WHITE if attacker_color == WHITE else chess.BLACK
        return self.board.is_attacked_by(c, sq)

    def is_in_check(self, color: Optional[int] = None) -> bool:
        if color is None or color == self.side_to_move:
            return self.board.is_check()
        c = chess.WHITE if color == WHITE else chess.BLACK
        k_sq = self.board.king(c)
        return False if k_sq is None else self.board.is_attacked_by(not c, k_sq)

    def legal_moves(self) -> List[Move]:
        return [
            Move(
                from_sq=mv.from_square,
                to_sq=mv.to_square,
                promotion=int(mv.promotion) if mv.promotion is not None else None,
            )
            for mv in self.board.legal_moves
        ]

    def step(self, move: Move) -> "Standard8x8State":
        b = self.board.copy(stack=False)
        b.push(chess.Move(move.from_sq, move.to_sq, promotion=move.promotion))
        return Standard8x8State(b)

    def step_by_edge_id(self, edge_id: int) -> "Standard8x8State":
        edge = self.graph.edges[edge_id]
        from_sq = (
            edge.from_sq if self.side_to_move == WHITE else self.graph.flip_rank_sq(edge.from_sq)
        )
        fx, fy = from_sq % 8, from_sq // 8
        is_pawn = int(self.piece_types[fy, fx]) == PAWN
        mv = self.graph.edge_id_to_move(edge_id, self.side_to_move, moving_piece_is_pawn=is_pawn)
        return self.step(mv)

    def legal_edge_indices(self) -> List[int]:
        return [self.graph.move_to_edge_id(mv, self.side_to_move) for mv in self.legal_moves()]

    def legal_edge_mask(self) -> np.ndarray:
        mask = np.zeros(self.graph.num_edges, dtype=bool)
        for idx in self.legal_edge_indices():
            mask[idx] = True
        return mask

    def is_checkmate(self) -> bool:
        return self.board.is_checkmate()

    def is_stalemate(self) -> bool:
        return self.board.is_stalemate()

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def mirror_vertical_swap_colors(self) -> "Standard8x8State":
        """Reflect board across horizontal midline (y -> 7 - y) and invert White <-> Black."""
        return Standard8x8State(self.board.mirror())


ChessState = Union[Gardner5x5State, Standard8x8State]

