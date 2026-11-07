# 01 — Mathematical and Engineering Tensions & Resolutions

**Date:** 2026-11-05  
**Status:** Active Architectural Specification (Refines Implementation Plan `v1.0`)  
**Tracked Alterations:** `ALT-001`, `ALT-002`, `ALT-003`, `ALT-004` (recorded in [`../flight-recorder.md`](../flight-recorder.md))

---

## Overview

While the three-pillar architecture (Topographical Move-Potential Fields + $\mathrm{SE}(2)$ Invariant Point Attention + D3PM Trajectory Diffusion) is conceptually sound, translating continuous geometric biology primitives (AlphaFold2 IPA) and continuous trajectory diffusion (Diffuser) to the discrete, anisotropic rules of chess introduces **four mathematical and engineering tensions**.

Addressing these tensions upfront prevents Stage-Gate failures in WP1–WP3 while preserving 100% compliance with the Stage-Gate verification metrics in the `v1.0` Implementation Plan.

---

## Tension 1: Continuous $\mathrm{SE}(2)$ Invariance vs. Chess Lattice Anisotropy & Board Boundaries (`ALT-001`)

### 1.1 Mathematical Formulation of the Problem

In AlphaFold2 ([Jumper et al., 2021]), proteins reside in unconstrained 3D Euclidean space $\mathbb{R}^3$ where physical energy is strictly invariant under global rigid-body transformations $g = (\mathbf{R}, \mathbf{t}) \in \mathrm{SE}(3)$. The `v1.0` specification proposes assigning each piece cluster $i$ a local frame $T_i = (R_i, \vec{x}_i) \in \mathrm{SE}(2)$ such that attention is invariant to global translations and rotations:

$$\mathbf{x}_i^{(j)} = R_i^\top (\vec{x}_j - \vec{x}_i)$$

However, the chessboard $B = \{0, \dots, 7\}^2 \subset \mathbb{Z}^2$ violates continuous $\mathrm{SE}(2)$ symmetry in three fundamental ways:

1. **Rank/File Anisotropy (Pawn Directionality):** White pawns advance strictly along $+y$ (increasing rank) and promote at $y=7$; Black pawns advance strictly along $-y$ and promote at $y=0$. A $90^\circ$ rotation $R(\pi/2) \in \mathrm{SO}(2)$ maps a vertical file into a horizontal rank, equating a passed pawn advancing toward promotion with a pawn moving sideways (impossible).
2. **Lattice Metric Incompatibility ($\ell_1$ vs. $\ell_\infty$ Rays):** On $\mathbb{Z}^2$, orthogonal Rook rays operate along $( \pm 1, 0), (0, \pm 1)$ (Euclidean step $1$, covering 14 squares from anywhere), whereas diagonal Bishop rays operate along $(\pm 1, \pm 1)$ (Euclidean step $\sqrt{2}$, restricted to a single square color complex and covering 7–13 squares depending on centrality). A $45^\circ$ rotation $R(\pi/4) \in \mathrm{SO}(2)$ maps Rook vectors onto Bishop vectors with a $\sqrt{2}$ metric distortion.
3. **Hard Boundary Blocking (Non-Translation Invariance):** Unlike $\mathbb{R}^2$, $B$ is a compact bounded lattice. The board edge blocks 3 to 5 king escape squares. A mating net or trapped piece cluster at $(x, y) = (0, 7)$ (corner `a8`) is frequently non-mating if translated by $\vec{t} = (+3, -3)$ to the center $(3, 4)$ (`d5`).

### 1.2 Architectural Resolution (`ALT-001`)

Instead of unconstrained continuous $\mathrm{SO}(2)$ rotations, we formulate the local reference frame $T_i = (R_i, \vec{x}_i)$ over the **Color-Aligned Dihedral Frame Group** coupled with **Boundary-Potential Anchors**:

1. **Color-Intrinsic Orientation $R_i \in \{I, M_y\} \times \mathbb{Z}_2$:**
   * Every active token $i$ belonging to side $c_i \in \{\text{White}, \text{Black}\}$ defines its local forward axis $\hat{e}_{\text{fwd}}(c_i) = (0, +1)$ for White and $(0, -1)$ for Black, and its lateral axis toward the kingside/queenside or board center.
   * Under simultaneous vertical board reflection ($y \mapsto 7 - y$) and color inversion ($\text{White} \leftrightarrow \text{Black}$), all relative local coordinates $\mathbf{x}_i^{(j)} = R_i^\top (\vec{x}_j - \vec{x}_i)$ remain **mathematically identical to machine precision**, guaranteeing the **Stage-Gate 2 Equivariance Metric** ($< 10^{-4}$ to $10^{-5}$ tolerance) by construction.
2. **Ray-Aligned Dihedral Sub-Frames ($D_4$):**
   * For directional piece rays (Rook orthogonal vs. Bishop diagonal), attention heads project relative displacement $(\Delta x, \Delta y) = \vec{x}_j - \vec{x}_i$ into separate **orthogonal ($\ell_1$)** and **diagonal ($\ell_\infty$)** invariant subspaces rather than mixing them via continuous $\mathrm{SO}(2)$ angles.
3. **Explicit Boundary-Potential Vector Field:**
   * To preserve boundary awareness under translation-invariant relative attention, each square $\vec{x}_i = (x_i, y_i)$ embeds a 4-component boundary proximity vector:
     $$\mathbf{b}(\vec{x}_i) = \left(\frac{x_i}{W-1},\; \frac{W-1-x_i}{W-1},\; \frac{y_i}{H-1},\; \frac{H-1-y_i}{H-1}\right) \in [0, 1]^4$$
   * This allows an IPA cluster to use translation-invariant relative attention weights while still conditioning on whether the local frame is adjacent to a board edge or corner.
4. **Castling Asymmetry Bias (Pre-emptive WP2 Fallback Integration):**
   * Incorporate the 4-bit castling-rights state and king-file orientation bias directly as a gated scalar offset in the IPA logits so asymmetric kingside/queenside castling geometry never degrades ranking accuracy.

---

## Tension 2: Static $G(V,E)$ Masking vs. State-Dependent Move Legality in Parallel D3PM (`ALT-002`)

### 2.1 Mathematical Formulation of the Problem

In WP3, the D3PM horizon planner synthesizes a 6-ply trajectory $\tau = (a_1, a_2, a_3, a_4, a_5, a_6)$ simultaneously by reversing an absorbing-state (`[MASK]`) Markov corruption process $Q_t$. Risk R2 in the `v1.0` plan specifies:
> *"Enforce a hard rule-based transition mask derived from legal move graphs ($G(V,E)$ with 1,858 valid edges) during the final 3 denoising steps."*

And Stage-Gate 3 requires:
> *"Legality Metric: The masked reverse diffusion process generates 100% legal move trajectories across 10,000 test positions."*

**The mathematical gap:**
* The static directed move graph $G(V, E)$ ($|E| = 1{,}858$ for 8x8 standard chess; $|E| = 455$ for 5x5 Gardner chess) defines the **action vocabulary** $\mathcal{V}_{\text{act}}$: the set of all moves that are geometrically possible on an empty board (including pawn promotions).
* However, at ply $k \in \{1, \dots, 6\}$, the set of actually legal moves $\mathcal{L}(s_{k-1}) \subset E$ is a strict, sparse subset (~30 edges out of 1,858) that depends deterministically on the **intermediate board state** $s_{k-1} = \mathrm{Step}(s_{k-2}, a_{k-1})$:
  $$\tau \text{ is legal} \iff \forall k \in \{1, \dots, 6\}, \quad a_k \in \mathcal{L}\big(\mathrm{Rollout}(s_0, a_{1:k-1})\big)$$
* Masking logits with the static graph $G(V, E)$ alone still permits ~1,828 **illegal** moves per ply (moving absent pieces, jumping over blockers, moving into check, ignoring check).

### 2.2 Architectural Resolution (`ALT-002`)

We implement a **Two-Tier Rollout-Projected Legality Mask** during reverse diffusion $t = T, \dots, 1$:

1. **Tier 1 — Static Graph Vocabulary Space ($t \in [1, T]$):**
   * Rather than diffusing over $64 \times 64 = 4{,}096$ raw square pairs and masking out unreachable geometries, the D3PM categorical state space at each ply $k$ is defined **directly over the valid edge index** $e \in \{0, \dots, |E|-1\} \cup \{\texttt{[MASK]}\}$ ($455+1$ tokens for 5x5 Gardner; $1{,}858+1$ tokens for 8x8 chess). Every non-mask token is guaranteed to be a valid edge in $G(V,E)$ at all diffusion steps.
2. **Tier 2 — Sequential State-Rollout Legality Projection ($t \le K_{\text{proj}}$, default $K_{\text{proj}} = 3$):**
   * At each of the final $K_{\text{proj}}$ denoising steps (and deterministically at $t = 1$), for a candidate trajectory distribution $p_\theta(a_{1:6} \mid \tilde{\tau}_t, z_0)$:
     1. Compute ply 1 legal mask $\mathbf{m}_1 = \mathbb{I}[e \in \mathcal{L}(s_0)]$, mask ply-1 logits $\tilde{\ell}_1 = \ell_1 + \log \mathbf{m}_1$, sample/select $\hat{a}_1$, and advance the board state $s_1 = \mathrm{Step}(s_0, \hat{a}_1)$.
     2. Inductively for $k = 2, \dots, 6$: compute $\mathbf{m}_k = \mathbb{I}[e \in \mathcal{L}(s_{k-1})]$, mask logits $\tilde{\ell}_k = \ell_k + \log \mathbf{m}_k$, select $\hat{a}_k$, and advance $s_k = \mathrm{Step}(s_{k-1}, \hat{a}_k)$ (handling terminal checkmate/stalemate states via a no-op `[PAD]`/`[TERM]` absorption token).
   * Because $p_\theta(\ell_{1:6} \mid \tilde{\tau}_t, z_0)$ is still evaluated **in a single non-autoregressive neural forward pass** (where ply $k+4$ features attend bidirectionally to ply $k+1$ features inside the transformer denoiser), this preserves non-directional horizon planning while guaranteeing **100.0% legal 6-ply move trajectories** across all test positions for Stage-Gate 3.
3. **Forced Mate-in-3 Puzzle Protocol:**
   * In forced mate-in-3 benchmarks (5 plies: $a_1, a_2, a_3, a_4, a_5\#$), the opponent may have multiple legal delaying responses at plies 2 and 4. We evaluate both (a) single-shot 6-ply Principal Variation (PV) synthesis and (b) interactive puzzle play where the D3PM planner re-synthesizes the remaining horizon if an external puzzle engine plays a non-PV legal defense.

---

## Tension 3: Superposition Ambiguity & Non-Spatial State Metadata in Reversible Topographical Fields (`ALT-003`)

### 3.1 Mathematical Formulation of the Problem

WP1 requires encoding each discrete board state $s$ into a continuous 2D scalar-vector Move-Potential Tensor $\mathcal{T} = \mathrm{Encode}(s) \in \mathbb{R}^{H \times W \times C}$ and implementing a deterministic decoder $\mathrm{Decode}(\mathcal{T})$ that satisfies **Stage-Gate 1**:
> *"Reconstruction Metric: The decoder reconstructs 100% of legal moves across 100,000 random test positions."*

If $\mathcal{T}$ consists solely of summed continuous influence fields $\Phi(\vec{x}) = \sum_{p} \phi_p(\vec{x})$, the mapping $s \mapsto \mathcal{T}$ is **not injective**:
1. **Superposition & Blocker Ambiguity (Risk R1):** Multiple piece configurations can produce overlapping ray intensities at interior squares, obscuring whether a square contains a pinned blocker or an empty ray intersection.
2. **Non-Spatial Markov Rules:** Exact legal move generation $\mathcal{L}(s)$ depends on state variables that do not exist in piece coordinates alone:
   * Side to move $c_{\text{stm}} \in \{0, 1\}$,
   * Castling rights $\mathbf{r}_{\text{castle}} \in \{0, 1\}^4$ (`WK`, `WQ`, `BK`, `BQ` — absent in 5x5 Gardner, required in 8x8),
   * En passant target square $sq_{\text{ep}} \in \{\varnothing\} \cup \{0, \dots, 63\}$,
   * Halfmove clock $h_{50} \in \mathbb{Z}_{\ge 0}$.

### 3.2 Architectural Resolution (`ALT-003`)

We construct the Topographical Move-Potential Tensor $\mathcal{T} \in \mathbb{R}^{H \times W \times C}$ as a **Bijective Structured Fiber Bundle** ($C = 44$ channels for 8x8; $C = 40$ for 5x5 Gardner) partitioned into three sub-bundles:

| Sub-Bundle | Channels | Description | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Continuous Topographical Potential Fields** | $24$ | • **Scalar Control & Contestation ($4$ ch):** White attack intensity, Black attack intensity, net control gradient ($I_W - I_B$), and non-linear contestation energy ($I_W \cdot I_B$).<br>• **Directional Ray Flux Vectors ($8$ ch):** Continuous $(\Delta x, \Delta y)$ unit-weighted flux along orthogonal and diagonal rays for each color, attenuated across blockers.<br>• **Piece-Specific Potential Fields ($6$ ch):** Knight jump manifold potentials, pawn capture threat cones, and king safety pressure gradients.<br>• **Tactical Tension & X-Ray Fields ($2$ ch):** Pin/discovered-attack alignment intensity along obstructed slider rays (Risk R1 mitigation).<br>• **Boundary Distance Potential ($4$ ch):** Normalized distance to N/S/E/W board rims (`ALT-001`). | Captures continuous topological interactions, wave interference, and pin/X-ray geometry natively in the input space. |
| **2. Discrete Source-Anchor Channels (Risk R1 Built-In)** | $13$ | • **Signed Piece Source Planes ($12$ ch):** 6 piece types (`P, N, B, R, Q, K`) $\times$ 2 colors (`White`, `Black`), representing the point-source emitters of the potential field.<br>• **Aggregate Occupancy Mask ($1$ ch):** Binary blocker mask $\mathbb{I}[\text{occupied}(x,y)]$ per Risk R1. | Guarantees zero ambiguity at ray intersections and provides exact source anchors for local IPA frames. |
| **3. Non-Spatial Markov Rule Planes** | $7$ (8x8)<br>$3$ (5x5) | • **Side to Move ($1$ ch):** Constant plane $\pm 1.0$.<br>• **Castling Rights ($4$ ch, 8x8 only):** Constant planes for `WK`, `WQ`, `BK`, `BQ`.<br>• **En Passant Target ($1$ ch, 8x8):** One-hot indicator at $sq_{\text{ep}}$ if a legal en passant capture exists.<br>• **Normalized Halfmove Clock ($1$ ch):** $h_{50} / 100.0$. | Makes $\mathrm{Decode}(\mathrm{Encode}(s)) = s$ strictly lossless ($100.0\%$ exact board & legal-move reconstruction on any valid FEN). |

---

## Tension 4: Stockfish `MultiPV` Q-Value Distillation Throughput, Depth Dilution & Determinism (`ALT-004`)

### 4.1 Mathematical Formulation of the Problem

WP1 specifies evaluating every sampled board position with Stockfish 16 for **50 milliseconds per position** and recording the **exact action-values (Q-values) for all legal moves** in each state.

In the UCI protocol, Stockfish only outputs evaluations for all $M = |\mathcal{L}(s)|$ legal root moves if invoked with `setoption name MultiPV value M` (where $M \approx 30$ in middle-game 8x8 positions). This creates two problems:
1. **Depth Dilution:** A 50ms wall-clock budget divided across $M = 30$ root variations via `MultiPV 30` leaves $< 1.7\text{ ms}$ per variation, reducing effective search depth and introducing noisy Q-values on non-top moves.
2. **Wall-Clock Non-Determinism (WP5 Conflict):** `go movetime 50` depends on instantaneous CPU frequency scaling, OS scheduling jitter, and background load. Running the pipeline twice on the same positions produces slightly different centipawn labels, violating WP5's deterministic reproducibility standard.

### 4.2 Architectural Resolution (`ALT-004`)

1. **Deterministic Fixed-Node Evaluation (`go nodes <N>`):**
   * Replace wall-clock `go movetime 50` with a **deterministic node budget** `go nodes N_budget` under `Threads = 1` per worker process (parallelized across independent worker processes, one per CPU core, with fixed `Hash` size and `Clear Hash` between uncorrelated positions).
   * In single-threaded Stockfish 16 with fixed `nodes`, the search tree and returned centipawn/mate scores are **100% deterministic and hardware-independent** across runs.
2. **Tiered `MultiPV` Allocation:**
   * For a position with $M$ legal moves, set `MultiPV = M` with a scaled node budget $N_{\text{budget}} = \max(N_{\text{base}},\, n_{\text{per\_move}} \cdot M)$ (e.g., $n_{\text{per\_move}} = 1{,}500$ nodes/move $\implies \sim 45{,}000$ nodes for $M=30$, taking ~25–35ms on a single Apple M4 Pro P-core, or $\sim 2.5\text{–}3.5\text{ ms}$ wall-clock per position across 10 P-cores).
3. **Win-Probability & Forced-Mate Calibration:**
   * Convert centipawn score $\text{cp}(s, a)$ into action-value $Q(s, a) \in [0, 1]$ via the calibrated Lichess/Stockfish logistic transform:
     $$Q(s, a) = \sigma\big(k_{\text{cp}} \cdot \text{cp}(s, a)\big) = \frac{1}{1 + \exp(-k_{\text{cp}} \cdot \text{cp}(s, a))}, \quad k_{\text{cp}} = 0.00368208$$
   * Map forced mate-in-$d$ ($d > 0$ for side to move) to $Q(s, a) = 1.0 - \epsilon_{\text{mate}} \cdot d$ ($\epsilon_{\text{mate}} = 10^{-4}$, preserving strict mate-distance ranking for Kendall $\tau$ while placing all forced mates in the top win-probability bin per WP1 Task 7), and mated-in-$|d|$ ($d < 0$) to $Q(s, a) = \epsilon_{\text{mate}} \cdot |d|$.
