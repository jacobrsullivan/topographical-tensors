"""WP1 Topographical Move-Potential Field Encoder & Deterministic Lossless Decoder (ALT-003).

Constructs a bijective 3-Bundle Move-Potential Tensor T in R^(C x H x W):
  - Bundle 1 (ch 0..23, 24 channels): Continuous Topographical Potential Fields
      ch[0]:    White local attack intensity I_W in [0, 1]
      ch[1]:    Black local attack intensity I_B in [0, 1]
      ch[2]:    Net control gradient (I_W - I_B) in [-1, 1]
      ch[3]:    Contestation energy tanh(4 * I_W * I_B) in [0, 1]
      ch[4:6]:  White orthogonal ray flux vector (Fx, Fy) in [-1, 1]
      ch[6:8]:  Black orthogonal ray flux vector (Fx, Fy) in [-1, 1]
      ch[8:10]: White diagonal ray flux vector (Fx, Fy) in [-1, 1]
      ch[10:12]:Black diagonal ray flux vector (Fx, Fy) in [-1, 1]
      ch[12]:   White knight jump potential field in [0, 1]
      ch[13]:   Black knight jump potential field in [0, 1]
      ch[14]:   White pawn capture & advance cone field in [0, 1]
      ch[15]:   Black pawn capture & advance cone field in [0, 1]
      ch[16]:   White King safety / attack pressure gradient in [-1, 1]
      ch[17]:   Black King safety / attack pressure gradient in [-1, 1]
      ch[18]:   White X-ray & pin alignment tension field (Risk R1) in [0, 1]
      ch[19]:   Black X-ray & pin alignment tension field (Risk R1) in [0, 1]
      ch[20:24]:Boundary distance potential (West, East, South, North) (ALT-001) in [0, 1]
  - Bundle 2 (ch 24..36, 13 channels): Discrete Source-Anchor & Blocker Planes (Risk R1)
      ch[24:30]:White piece source planes (P, N, B, R, Q, K) in {0, 1}
      ch[30:36]:Black piece source planes (p, n, b, r, q, k) in {0, 1}
      ch[36]:   Aggregate blocker occupancy mask in {0, 1}
  - Bundle 3 (ch 37..39 for 5x5 [C=40]; ch 37..43 for 8x8 [C=44]): Markov Rule-State Planes
      ch[37]:   Side to move (+1.0 for White, -1.0 for Black)
      ch[38]:   Normalized halfmove clock min(halfmove_clock / 100.0, 1.0)
      (For 5x5 Gardner):
      ch[39]:   Normalized material density scalar
      (For 8x8 Standard):
      ch[39]:   White Kingside castling right (WK) in {0, 1}
      ch[40]:   White Queenside castling right (WQ) in {0, 1}
      ch[41]:   Black Kingside castling right (BK) in {0, 1}
      ch[42]:   Black Queenside castling right (BQ) in {0, 1}
      ch[43]:   En passant target square one-hot indicator in {0, 1}
"""

from __future__ import annotations

from typing import Union
import chess
import numpy as np

from topo_chess.domains.boards import Gardner5x5State, Standard8x8State
from topo_chess.domains.move_graph import (
    BISHOP,
    BLACK,
    EMPTY,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
)

ChessState = Union[Gardner5x5State, Standard8x8State]

# Relative attacker force weights (inverse piece vulnerability / offensive projection weight)
PIECE_FORCE_WEIGHT = {
    PAWN: 0.35,
    KNIGHT: 0.30,
    BISHOP: 0.30,
    ROOK: 0.25,
    QUEEN: 0.22,
    KING: 0.18,
}

PIECE_TARGET_VALUE = {
    PAWN: 0.15,
    KNIGHT: 0.35,
    BISHOP: 0.35,
    ROOK: 0.55,
    QUEEN: 0.90,
    KING: 1.00,
}


class TopographicalEncoder:
    """Encodes a discrete ChessState into a continuous 3-bundle Move-Potential Tensor."""

    def __init__(self, height: int = 8, width: int = 8) -> None:
        self.height = height
        self.width = width
        self.is_8x8 = height == 8 and width == 8
        self.num_channels = 44 if self.is_8x8 else 40

        # Precompute static boundary proximity potentials (ALT-001) and coordinate grids
        self.boundary_planes = np.zeros((4, height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                self.boundary_planes[0, y, x] = x / max(width - 1, 1)  # West
                self.boundary_planes[1, y, x] = (width - 1 - x) / max(width - 1, 1)  # East
                self.boundary_planes[2, y, x] = y / max(height - 1, 1)  # South
                self.boundary_planes[3, y, x] = (height - 1 - y) / max(height - 1, 1)  # North
        self._yy, self._xx = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )

    def encode(self, state: ChessState) -> np.ndarray:
        """Encode board state into a (C, H, W) float32 Move-Potential Tensor in global board coordinates."""
        h, w = self.height, self.width
        tensor = np.zeros((self.num_channels, h, w), dtype=np.float32)

        pt = state.piece_types
        pc = state.piece_colors
        max_step = max(h, w) - 1

        # Bundle 2: Discrete piece source planes & aggregate occupancy (ch 24..36)
        for p_type in range(1, 7):
            tensor[24 + (p_type - 1)] = (pt == p_type) & (pc == WHITE)
            tensor[30 + (p_type - 1)] = (pt == p_type) & (pc == BLACK)
        tensor[36] = (pt != EMPTY).astype(np.float32)

        # Bundle 1: Continuous Topographical Potential Fields (ch 0..23)
        attack_sum = np.zeros((2, h, w), dtype=np.float32)
        orth_flux = np.zeros((2, 2, h, w), dtype=np.float32)  # [color, (fx, fy), y, x]
        diag_flux = np.zeros((2, 2, h, w), dtype=np.float32)  # [color, (fx, fy), y, x]
        knight_pot = np.zeros((2, h, w), dtype=np.float32)
        pawn_pot = np.zeros((2, h, w), dtype=np.float32)
        xray_pot = np.zeros((2, h, w), dtype=np.float32)

        king_pos = {WHITE: None, BLACK: None}

        for fy in range(h):
            for fx in range(w):
                p = int(pt[fy, fx])
                if p == EMPTY:
                    continue
                c = int(pc[fy, fx])
                w_force = PIECE_FORCE_WEIGHT[p]

                if p == KING:
                    king_pos[c] = (fx, fy)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            tx, ty = fx + dx, fy + dy
                            if 0 <= tx < w and 0 <= ty < h:
                                attack_sum[c, ty, tx] += w_force

                elif p == PAWN:
                    p_dir = 1 if c == WHITE else -1
                    ty = fy + p_dir
                    if 0 <= ty < h:
                        # Forward advance potential
                        if pt[ty, fx] == EMPTY:
                            pawn_pot[c, ty, fx] += 0.35
                        # Diagonal capture threat cones
                        for dx in (-1, 1):
                            tx = fx + dx
                            if 0 <= tx < w:
                                attack_sum[c, ty, tx] += w_force
                                pawn_pot[c, ty, tx] += 0.65

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
                        if 0 <= tx < w and 0 <= ty < h:
                            attack_sum[c, ty, tx] += w_force
                            knight_pot[c, ty, tx] += 0.50

                else:
                    # Sliding pieces (Bishop, Rook, Queen) with ray attenuation & X-ray/pin tracking
                    ray_list = []
                    if p in (ROOK, QUEEN):
                        ray_list.extend([(1, 0, True), (-1, 0, True), (0, 1, True), (0, -1, True)])
                    if p in (BISHOP, QUEEN):
                        ray_list.extend(
                            [(1, 1, False), (1, -1, False), (-1, 1, False), (-1, -1, False)]
                        )

                    for dx, dy, is_orth in ray_list:
                        norm = float(np.hypot(dx, dy))
                        ux, uy = dx / norm, dy / norm
                        blockers_seen = 0
                        first_blocker_pos = None

                        for step in range(1, max_step + 1):
                            tx, ty = fx + dx * step, fy + dy * step
                            if not (0 <= tx < w and 0 <= ty < h):
                                break
                            atten = float(1.0 - 0.08 * (step - 1))
                            target_p = int(pt[ty, tx])

                            if blockers_seen == 0:
                                attack_sum[c, ty, tx] += w_force * atten
                                if is_orth:
                                    orth_flux[c, 0, ty, tx] += ux * w_force * atten
                                    orth_flux[c, 1, ty, tx] += uy * w_force * atten
                                else:
                                    diag_flux[c, 0, ty, tx] += ux * w_force * atten
                                    diag_flux[c, 1, ty, tx] += uy * w_force * atten

                                if target_p != EMPTY:
                                    blockers_seen = 1
                                    first_blocker_pos = (tx, ty)
                            elif blockers_seen == 1:
                                # X-ray projection through a single blocker (pins & discovered attacks)
                                xray_pot[c, ty, tx] += 0.35 * w_force * atten
                                if target_p != EMPTY:
                                    target_c = int(pc[ty, tx])
                                    if target_c != c and first_blocker_pos is not None:
                                        bx, by = first_blocker_pos
                                        # Highlight the pinned/discovered blocker square as well
                                        t_val = PIECE_TARGET_VALUE[target_p]
                                        xray_pot[c, by, bx] += 0.75 * t_val
                                        xray_pot[c, ty, tx] += 0.75 * t_val
                                    break

        I_W = np.tanh(attack_sum[WHITE])
        I_B = np.tanh(attack_sum[BLACK])
        tensor[0] = I_W
        tensor[1] = I_B
        tensor[2] = I_W - I_B
        tensor[3] = np.tanh(4.0 * I_W * I_B)

        tensor[4:6] = np.tanh(orth_flux[WHITE])
        tensor[6:8] = np.tanh(orth_flux[BLACK])
        tensor[8:10] = np.tanh(diag_flux[WHITE])
        tensor[10:12] = np.tanh(diag_flux[BLACK])

        tensor[12] = np.clip(knight_pot[WHITE], 0.0, 1.0)
        tensor[13] = np.clip(knight_pot[BLACK], 0.0, 1.0)
        tensor[14] = np.clip(pawn_pot[WHITE], 0.0, 1.0)
        tensor[15] = np.clip(pawn_pot[BLACK], 0.0, 1.0)

        # King safety gradients (ch 16..17): Gaussian-weighted net control around each King
        for c, ch_idx in ((WHITE, 16), (BLACK, 17)):
            k_xy = king_pos[c]
            if k_xy is not None:
                kx, ky = k_xy
                dist_sq = (self._xx - kx) ** 2 + (self._yy - ky) ** 2
                kernel = np.exp(-dist_sq / 3.0)
                friendly_I = I_W if c == WHITE else I_B
                enemy_I = I_B if c == WHITE else I_W
                tensor[ch_idx] = np.clip(kernel * (friendly_I - enemy_I), -1.0, 1.0)

        tensor[18] = np.clip(xray_pot[WHITE], 0.0, 1.0)
        tensor[19] = np.clip(xray_pot[BLACK], 0.0, 1.0)
        tensor[20:24] = self.boundary_planes

        # Bundle 3: Non-spatial Markov rule planes
        tensor[37] = 1.0 if state.side_to_move == WHITE else -1.0
        tensor[38] = min(float(state.halfmove_clock) / 100.0, 1.0)

        if self.is_8x8:
            c_mask = state.castling_mask
            tensor[39] = 1.0 if (c_mask & 1) else 0.0  # WK
            tensor[40] = 1.0 if (c_mask & 2) else 0.0  # WQ
            tensor[41] = 1.0 if (c_mask & 4) else 0.0  # BK
            tensor[42] = 1.0 if (c_mask & 8) else 0.0  # BQ
            if state.ep_square is not None:
                ep_x, ep_y = state.ep_square % w, state.ep_square // w
                tensor[43, ep_y, ep_x] = 1.0
        else:
            tensor[39] = float(np.count_nonzero(pt != EMPTY)) / float(h * w)

        return tensor

    def encode_canonical(self, state: ChessState) -> np.ndarray:
        """Encode board state in the active player's color-aligned canonical frame (ALT-001).

        If Black is to move, equivalences state with its vertically mirrored & color-swapped twin
        so that encode_canonical(s) == encode_canonical(s.mirror_vertical_swap_colors()).
        """
        canon_state = state if state.side_to_move == WHITE else state.mirror_vertical_swap_colors()
        return self.encode(canon_state)


class TopographicalDecoder:
    """Deterministic decoder that reconstructs the exact discrete ChessState from a Move-Potential Tensor."""

    def __init__(self, height: int = 8, width: int = 8) -> None:
        self.height = height
        self.width = width
        self.is_8x8 = height == 8 and width == 8

    def decode(self, tensor: np.ndarray) -> ChessState:
        """Reconstruct exact Gardner5x5State or Standard8x8State from encoded (C, H, W) tensor."""
        h, w = self.height, self.width
        pt = np.zeros((h, w), dtype=np.int8)
        pc = np.zeros((h, w), dtype=np.int8)

        for p_type in range(1, 7):
            w_mask = tensor[24 + (p_type - 1)] > 0.5
            b_mask = tensor[30 + (p_type - 1)] > 0.5
            pt[w_mask] = p_type
            pc[w_mask] = WHITE
            pt[b_mask] = p_type
            pc[b_mask] = BLACK

        stm = WHITE if float(tensor[37, 0, 0]) >= 0.0 else BLACK
        hm_clock = int(round(float(tensor[38, 0, 0]) * 100.0))

        if not self.is_8x8:
            return Gardner5x5State(
                piece_types=pt,
                piece_colors=pc,
                side_to_move=stm,
                halfmove_clock=hm_clock,
                fullmove_number=1,
            )

        board = chess.Board(None)
        board.turn = chess.WHITE if stm == WHITE else chess.BLACK
        board.halfmove_clock = hm_clock
        board.fullmove_number = 1

        for y in range(8):
            for x in range(8):
                p = int(pt[y, x])
                if p != EMPTY:
                    c = chess.WHITE if int(pc[y, x]) == WHITE else chess.BLACK
                    board.set_piece_at(y * 8 + x, chess.Piece(p, c))

        castling_rights = chess.BB_EMPTY
        if float(tensor[39, 0, 0]) > 0.5:
            castling_rights |= chess.BB_H1
        if float(tensor[40, 0, 0]) > 0.5:
            castling_rights |= chess.BB_A1
        if float(tensor[41, 0, 0]) > 0.5:
            castling_rights |= chess.BB_H8
        if float(tensor[42, 0, 0]) > 0.5:
            castling_rights |= chess.BB_A8
        board.castling_rights = castling_rights

        ep_plane = tensor[43]
        if float(np.max(ep_plane)) > 0.5:
            ep_idx = int(np.argmax(ep_plane))
            board.ep_square = ep_idx
        else:
            board.ep_square = None

        return Standard8x8State(board)
