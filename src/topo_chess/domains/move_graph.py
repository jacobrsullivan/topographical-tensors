"""Exact Move Graph G(V, E) definitions for 5x5 Gardner (455 edges) and 8x8 Standard Chess (1,858 edges).

Mathematical Derivation of Canonical Edge Counts:
- Moves are represented in the active player's canonical orientation (White moves +y from rank 0 to H-1;
  Black moves are vertically mirrored y -> H - 1 - y so the active player always advances toward rank H-1).
- On an H x W board:
  1. Queen-like rays (4 orthogonal + 4 diagonal directions, steps 1..max(H,W)-1, including default Queen promotions):
     - 5x5 Gardner:  200 orthogonal + 120 diagonal = 320 edges
     - 8x8 Standard: 896 orthogonal + 560 diagonal = 1,456 edges
  2. Knight jumps (8 L-shaped offsets (±1, ±2), (±2, ±1)):
     - 5x5 Gardner:  96 edges
     - 8x8 Standard: 336 edges
  3. Underpromotions (from rank y = H - 2 to rank y = H - 1 via Δx ∈ {-1, 0, +1} into piece ∈ {N, B, R}):
     - 5x5 Gardner:  (5 pushes + 8 captures) * 3 pieces = 13 * 3 = 39 edges
     - 8x8 Standard: (8 pushes + 14 captures) * 3 pieces = 22 * 3 = 66 edges
  Total Valid Edges |E|:
     - 5x5 Gardner:  320 + 96 + 39 = 455 valid edges
     - 8x8 Standard: 1,456 + 336 + 66 = 1,858 valid edges
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Piece type constants
EMPTY = 0
PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

WHITE = 0
BLACK = 1

UNDERPROMOTION_PIECES = (KNIGHT, BISHOP, ROOK)
ALL_PROMOTION_PIECES = (QUEEN, ROOK, BISHOP, KNIGHT)


@dataclass(frozen=True)
class Move:
    """Discrete chess move in board coordinates (0-indexed: file x in 0..W-1, rank y in 0..H-1)."""

    from_sq: int
    to_sq: int
    promotion: Optional[int] = None  # None or one of (KNIGHT, BISHOP, ROOK, QUEEN)

    def to_uci(self, width: int = 8) -> str:
        fx, fy = self.from_sq % width, self.from_sq // width
        tx, ty = self.to_sq % width, self.to_sq // width
        promo_map = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}
        suffix = promo_map.get(self.promotion, "") if self.promotion is not None else ""
        return f"{chr(ord('a') + fx)}{fy + 1}{chr(ord('a') + tx)}{ty + 1}{suffix}"

    @classmethod
    def from_uci(cls, uci: str, width: int = 8) -> "Move":
        fx = ord(uci[0]) - ord("a")
        fy = int(uci[1]) - 1
        tx = ord(uci[2]) - ord("a")
        ty = int(uci[3]) - 1
        promo_rev = {"n": KNIGHT, "b": BISHOP, "r": ROOK, "q": QUEEN}
        promo = promo_rev[uci[4].lower()] if len(uci) > 4 else None
        return cls(from_sq=fy * width + fx, to_sq=ty * width + tx, promotion=promo)


@dataclass(frozen=True)
class CanonicalEdge:
    """A directed edge in the canonical side-to-move move graph G(V, E)."""

    edge_id: int
    from_sq: int
    to_sq: int
    dx: int
    dy: int
    is_knight: bool
    is_orthogonal: bool
    is_diagonal: bool
    underpromotion: Optional[int] = None  # None (includes default QUEEN promo) or KNIGHT/BISHOP/ROOK


class MoveGraph:
    """Bijective mapping between legal chess moves and canonical graph edge indices e in 0..|E|-1."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.num_squares = height * width
        self.edges: List[CanonicalEdge] = []
        self._key_to_edge: Dict[Tuple[int, int, Optional[int]], int] = {}
        self._build_graph()

        # Precompute numpy arrays for fast vectorized neural indexing
        self.edge_from_sq = np.array([e.from_sq for e in self.edges], dtype=np.int64)
        self.edge_to_sq = np.array([e.to_sq for e in self.edges], dtype=np.int64)
        self.edge_dx = np.array([e.dx for e in self.edges], dtype=np.float32)
        self.edge_dy = np.array([e.dy for e in self.edges], dtype=np.float32)
        self.edge_type = np.array(
            [
                3
                if e.underpromotion is not None
                else (2 if e.is_knight else (1 if e.is_diagonal else 0))
                for e in self.edges
            ],
            dtype=np.int64,
        )
        self.edge_underpromo = np.array(
            [0 if e.underpromotion is None else e.underpromotion for e in self.edges],
            dtype=np.int64,
        )

    def _build_graph(self) -> None:
        h, w = self.height, self.width
        max_step = max(h, w) - 1
        ray_dirs = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        knight_dirs = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ]

        edge_id = 0
        # 1. Queen-like rays (orthogonal + diagonal)
        for fy in range(h):
            for fx in range(w):
                from_sq = fy * w + fx
                for dx, dy in ray_dirs:
                    for step in range(1, max_step + 1):
                        tx, ty = fx + dx * step, fy + dy * step
                        if 0 <= tx < w and 0 <= ty < h:
                            to_sq = ty * w + tx
                            is_orth = dx == 0 or dy == 0
                            edge = CanonicalEdge(
                                edge_id=edge_id,
                                from_sq=from_sq,
                                to_sq=to_sq,
                                dx=tx - fx,
                                dy=ty - fy,
                                is_knight=False,
                                is_orthogonal=is_orth,
                                is_diagonal=not is_orth,
                                underpromotion=None,
                            )
                            self.edges.append(edge)
                            self._key_to_edge[(from_sq, to_sq, None)] = edge_id
                            edge_id += 1

        # 2. Knight jumps
        for fy in range(h):
            for fx in range(w):
                from_sq = fy * w + fx
                for dx, dy in knight_dirs:
                    tx, ty = fx + dx, fy + dy
                    if 0 <= tx < w and 0 <= ty < h:
                        to_sq = ty * w + tx
                        edge = CanonicalEdge(
                            edge_id=edge_id,
                            from_sq=from_sq,
                            to_sq=to_sq,
                            dx=dx,
                            dy=dy,
                            is_knight=True,
                            is_orthogonal=False,
                            is_diagonal=False,
                            underpromotion=None,
                        )
                        self.edges.append(edge)
                        self._key_to_edge[(from_sq, to_sq, None)] = edge_id
                        edge_id += 1

        # 3. Underpromotions (from rank H-2 to H-1 for canonical active player)
        fy = h - 2
        ty = h - 1
        for fx in range(w):
            from_sq = fy * w + fx
            for dx in (-1, 0, 1):
                tx = fx + dx
                if 0 <= tx < w:
                    to_sq = ty * w + tx
                    for promo_piece in UNDERPROMOTION_PIECES:
                        edge = CanonicalEdge(
                            edge_id=edge_id,
                            from_sq=from_sq,
                            to_sq=to_sq,
                            dx=dx,
                            dy=1,
                            is_knight=False,
                            is_orthogonal=(dx == 0),
                            is_diagonal=(dx != 0),
                            underpromotion=promo_piece,
                        )
                        self.edges.append(edge)
                        self._key_to_edge[(from_sq, to_sq, promo_piece)] = edge_id
                        edge_id += 1

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def theoretical_actions(self) -> int:
        """Theoretical unconstrained square-to-square action count (N^2 * promos or N*(N-1))."""
        if self.height == 5 and self.width == 5:
            # 25 * 49 = 1,225 theoretical actions in 5x5 Gardner spec
            return 1225
        return 4096

    def flip_rank_sq(self, sq: int) -> int:
        """Reflect square vertically across horizontal board midline (rank y -> H - 1 - y)."""
        x = sq % self.width
        y = sq // self.width
        return (self.height - 1 - y) * self.width + x

    def move_to_edge_id(self, move: Move, side_to_move: int = WHITE) -> int:
        """Map a concrete board Move to its canonical edge index in 0..|E|-1."""
        from_sq = move.from_sq if side_to_move == WHITE else self.flip_rank_sq(move.from_sq)
        to_sq = move.to_sq if side_to_move == WHITE else self.flip_rank_sq(move.to_sq)
        underpromo = move.promotion if move.promotion in UNDERPROMOTION_PIECES else None
        return self._key_to_edge[(from_sq, to_sq, underpromo)]

    def edge_id_to_move(
        self, edge_id: int, side_to_move: int = WHITE, moving_piece_is_pawn: bool = False
    ) -> Move:
        """Convert a canonical edge index back to a board Move for side_to_move."""
        edge = self.edges[edge_id]
        from_sq = edge.from_sq if side_to_move == WHITE else self.flip_rank_sq(edge.from_sq)
        to_sq = edge.to_sq if side_to_move == WHITE else self.flip_rank_sq(edge.to_sq)

        promo: Optional[int] = edge.underpromotion
        if promo is None and moving_piece_is_pawn:
            canon_to_rank = edge.to_sq // self.width
            if canon_to_rank == self.height - 1:
                promo = QUEEN
        return Move(from_sq=from_sq, to_sq=to_sq, promotion=promo)


GARDNER_5X5_GRAPH = MoveGraph(height=5, width=5)
STANDARD_8X8_GRAPH = MoveGraph(height=8, width=8)
