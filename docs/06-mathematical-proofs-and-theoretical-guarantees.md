# 06 — Formal Mathematical Proofs & Theoretical Guarantees (`WP5 / v2.0`)

**Date:** 2026-11-06  
**Work Package:** WP5 (Reproducibility Audit & Editorial Review)  
**Verified Implementation Modules:**
* [`src/topo_chess/encoding/topographical_encoder.py`](../src/topo_chess/encoding/topographical_encoder.py)
* [`src/topo_chess/models/ipa_encoder.py`](../src/topo_chess/models/ipa_encoder.py)
* [`src/topo_chess/diffusion/d3pm_planner.py`](../src/topo_chess/diffusion/d3pm_planner.py)
* [`src/topo_chess/evaluation/gauntlet_and_bayeselo.py`](../src/topo_chess/evaluation/gauntlet_and_bayeselo.py)

---

## 1. Theorem 1: Bijectivity & Lossless Legal-Move Reconstruction of the 3-Bundle Move-Potential Tensor (`ALT-003`)

### 1.1 Formal Statement
Let $\mathcal{S}_{H \times W}$ denote the discrete state space of legal chess configurations on an $H \times W$ board ($(H, W) \in \{(5, 5), (8, 8)\}$), where each state $s = (\mathbf{P}, \mathbf{C}, c_{\text{stm}}, \mathbf{r}_{\text{castle}}, e_{\text{ep}}, h_{50}) \in \mathcal{S}_{H \times W}$ consists of:
1. Piece type grid $\mathbf{P} \in \{0, 1, \dots, 6\}^{H \times W}$ and color grid $\mathbf{C} \in \{0, 1\}^{H \times W}$,
2. Active side-to-move $c_{\text{stm}} \in \{0, 1\}$,
3. Castling availability bitmask $\mathbf{r}_{\text{castle}} \in \{0, 1\}^4$ (for $8 \times 8$; $\mathbf{0}$ for $5 \times 5$),
4. En-passant target square $e_{\text{ep}} \in \{0, \dots, HW-1\} \cup \{\varnothing\}$ (for $8 \times 8$), and
5. Normalized halfmove clock $h_{50} \in [0, 1]$.

Let $\Phi: \mathcal{S}_{H \times W} \to \mathbb{R}^{C \times H \times W}$ ($C = 44$ for $8 \times 8$; $C = 40$ for $5 \times 5$) be the **3-Bundle Topographical Move-Potential Encoder** defined in [`TopographicalEncoder.encode`](../src/topo_chess/encoding/topographical_encoder.py#L64-L214), and let $\Psi: \mathbb{R}^{C \times H \times W} \to \mathcal{S}_{H \times W}$ be the deterministic algebraic decoder defined in [`TopographicalDecoder.decode`](../src/topo_chess/encoding/topographical_encoder.py#L244-L315).

**Theorem 1.** *The encoding map $\Phi$ is strictly injective on $\mathcal{S}_{H \times W}$, and $\Psi$ is a left-inverse of $\Phi$: for every $s \in \mathcal{S}_{H \times W}$,*
$$\Psi(\Phi(s)) = s \quad \text{and} \quad \mathcal{L}(\Psi(\Phi(s))) = \mathcal{L}(s),$$
*where $\mathcal{L}(s) \subset E$ is the set of strictly legal moves in state $s$. Consequently, the Jaccard similarity $J(\mathcal{L}(\Psi(\Phi(s))), \mathcal{L}(s)) = 1.0000$ across all $s \in \mathcal{S}_{H \times W}$.*

### 1.2 Proof
1. **Non-Injectivity of Superposed Continuous Fields Alone:**
   Bundle 1 ($\mathcal{T}_{0:23} \in [-1, 1]^{24 \times H \times W}$) computes continuous ray-projection potentials, directional flux vectors $(\vec{F}_x, \vec{F}_y)$, local attack/defense energy $E_{\text{atk}} - E_{\text{def}}$, pin/X-ray alignment tension, and boundary distance potentials. Because continuous ray fields sum contributions over multiple piece sources at intersection squares, the restriction $\Phi|_{0:23}$ is not injective when two collinear sliding pieces share a ray or when castling/en-passant history modifies move legality without changing board occupancy.
2. **Exact Recovery of $(\mathbf{P}, \mathbf{C})$ from Bundle 2 ($\mathcal{T}_{24:36}$):**
   Channels $24 \dots 29$ encode one-hot indicator planes for White pieces $p \in \{1, \dots, 6\}$, and channels $30 \dots 35$ encode one-hot indicator planes for Black pieces $p \in \{1, \dots, 6\}$:
   $$\mathcal{T}_{23+p + 6c}(y, x) = \mathbb{I}[\mathbf{P}(y, x) = p \wedge \mathbf{C}(y, x) = c].$$
   Because at most one piece occupies any square $(y, x)$, the 12-vector $\mathcal{T}_{24:35}(y, x)$ has norm $0$ if and only if $\mathbf{P}(y, x) = 0$ (confirmed by blocker plane $\mathcal{T}_{36}(y, x) = 0$), and has a unique unit entry at index $k^* = \arg\max_{k \in \{0..11\}} \mathcal{T}_{24+k}(y, x) > 0.5$ otherwise, recovering $\mathbf{P}(y, x) = (k^* \bmod 6) + 1$ and $\mathbf{C}(y, x) = \lfloor k^* / 6 \rfloor$ without error.
3. **Exact Recovery of Markov Rule Metadata from Bundle 3 ($\mathcal{T}_{37:C-1}$):**
   Channel $\mathcal{T}_{37}$ stores the constant plane $c_{\text{stm}} \in \{0.0, 1.0\}$; channel $\mathcal{T}_{38}$ stores $h_{50} = \text{halfmove\_clock} / 100.0$. For $8 \times 8$, channels $\mathcal{T}_{39:42}$ store the 4 binary castling flags $\mathbf{r}_{\text{castle}} \in \{0.0, 1.0\}^4$, and channel $\mathcal{T}_{43}$ stores the one-hot indicator $\mathbb{I}[yW + x = e_{\text{ep}}]$. Reading $\mathcal{T}_{37:C-1}$ via threshold $0.5$ (and `round(100 * T[38])`) uniquely reconstructs $(c_{\text{stm}}, \mathbf{r}_{\text{castle}}, e_{\text{ep}}, h_{50})$.
4. **Invariance of the Legal Move Set $\mathcal{L}(s)$:**
   Because the rules of chess (and 5x5 Gardner chess) are a deterministic function of $s = (\mathbf{P}, \mathbf{C}, c_{\text{stm}}, \mathbf{r}_{\text{castle}}, e_{\text{ep}}, h_{50})$ and $\Psi(\Phi(s)) = s$ holds identically, $\mathcal{L}(\Psi(\Phi(s))) = \mathcal{L}(s)$ for all $s \in \mathcal{S}_{H \times W}$. $\blacksquare$

* **Empirical Verification (`SG1-C`):** Verified across `100,000` random 5x5 and 8x8 positions on `apple-m4-workstation` (`100,000 / 100,000` exact Zobrist and legal-move matches, `1.0000` Jaccard index).

---

## 2. Theorem 2: Exact Vertical-Mirror + Color-Inversion Equivariance of Color-Aligned Dihedral Local-Frame IPA (`ALT-001`)

### 2.1 Formal Statement
Define the **board-mirror and color-inversion involution** $\mathcal{M}: \mathcal{S}_{H \times W} \to \mathcal{S}_{H \times W}$ by:
1. Vertical rank reflection $y \mapsto \tilde{y} = H - 1 - y$ (with file $x \mapsto x$ preserved),
2. Piece color swap $c \mapsto \tilde{c} = 1 - c$ and active player swap $c_{\text{stm}} \mapsto 1 - c_{\text{stm}}$,
3. White/Black castling rights swap $(K, Q, k, q) \mapsto (k, q, K, Q)$, and
4. En-passant target reflection $(x_{\text{ep}}, y_{\text{ep}}) \mapsto (x_{\text{ep}}, H - 1 - y_{\text{ep}})$.

In the canonical side-to-move move graph $G(V, E)$, every physical move $m = ((x_1, y_1) \to (x_2, y_2), p_{\text{promo}})$ played by color $c_{\text{stm}}$ is mapped to its canonical edge index $e = \gamma(m, c_{\text{stm}}) \in \{0, \dots, |E|-1\}$ by orienting ranks relative to the active player's forward axis:
$$y_{\text{can}}(y, c_{\text{stm}}) = \begin{cases} y & \text{if } c_{\text{stm}} = 0 \text{ (White)}, \\ H - 1 - y & \text{if } c_{\text{stm}} = 1 \text{ (Black)}. \end{cases}$$

**Theorem 2.** *Let $f_\theta: \mathcal{S}_{H \times W} \to [0, 1]^{|E|}$ be the [`TopographicalIPAEncoder`](../src/topo_chess/models/ipa_encoder.py#L215-L375) action-value network composed with canonical encoding $\Phi_{\text{can}}(s) = \Phi(s_{\text{can}})$. For every state $s \in \mathcal{S}_{H \times W}$ and every canonical edge $e \in \{0, \dots, |E|-1\}$,*
$$f_\theta(\mathcal{M}(s))_e = f_\theta(s)_e.$$

### 2.2 Proof
1. **Invariance of the Canonical State Transformation:**
   Let $\mathcal{C}(s)$ denote the canonicalization operator implemented in [`TopographicalEncoder.encode_canonical`](../src/topo_chess/encoding/topographical_encoder.py#L216-L241):
   $$\mathcal{C}(s) = \begin{cases} s & \text{if } c_{\text{stm}}(s) = 0, \\ \mathcal{M}(s) & \text{if } c_{\text{stm}}(s) = 1. \end{cases}$$
   Because $\mathcal{M}$ is an involution ($\mathcal{M}(\mathcal{M}(s)) = s$) that flips $c_{\text{stm}} \mapsto 1 - c_{\text{stm}}$:
   * If $c_{\text{stm}}(s) = 0$, then $c_{\text{stm}}(\mathcal{M}(s)) = 1$, so $\mathcal{C}(\mathcal{M}(s)) = \mathcal{M}(\mathcal{M}(s)) = s = \mathcal{C}(s)$.
   * If $c_{\text{stm}}(s) = 1$, then $c_{\text{stm}}(\mathcal{M}(s)) = 0$, so $\mathcal{C}(\mathcal{M}(s)) = \mathcal{M}(s) = \mathcal{C}(s)$.
   Thus $\mathcal{C}(\mathcal{M}(s)) = \mathcal{C}(s)$ holds identically for all $s \in \mathcal{S}_{H \times W}$.
2. **Well-Posedness of Local $\mathrm{SE}(2)$/Dihedral Reference Frames on Canonical Tokens:**
   For each canonical square $i \in \{0, \dots, HW-1\}$ with normalized canonical coordinates $\vec{x}_i^0 = (x_i / (W-1), y_i / (H-1)) \in [0, 1]^2$, [`TopographicalIPAEncoder`](../src/topo_chess/models/ipa_encoder.py#L338-L339) predicts a local affine frame origin:
   $$\vec{o}_i(s) = \vec{x}_i^0 + 0.1 \tanh(W_{\text{shift}} \Phi(\mathcal{C}(s))_i), \quad R_i = I_2 \text{ (aligned with } \hat{e}_{\text{fwd}} = +y_{\text{can}}\text{)}.$$
   In each [`SparseLocalFrameIPA`](../src/topo_chess/models/ipa_encoder.py#L39-L185) layer, query and key points $\vec{q}_{i,h,p}, \vec{k}_{j,h,p} \in \mathbb{R}^2$ are transformed into global canonical coordinates via $T_i \circ \vec{q}_{i,h,p} = \vec{o}_i + \vec{q}_{i,h,p}$ in `torch.float32`, and the attention logit between squares $(i, j)$ is:
   $$A_{i,j}^{(h)} = \frac{\langle \mathbf{q}_{i,h}, \mathbf{k}_{j,h} \rangle}{\sqrt{d_k}} - \frac{w_h}{2} \sum_{p=1}^{P} \|(\vec{o}_i + \vec{q}_{i,h,p}) - (\vec{o}_j + \vec{k}_{j,h,p})\|_2^2 + \phi_h(\mathbf{g}_{i,j}) + 0.05 \psi_h(\mathbf{r}_{\text{rule}}),$$
   masked to the sparse topological adjacency graph $\mathcal{E}_{\text{sparse}}$ (orthogonal rays, diagonal rays, knight jumps, and 1-hop King neighborhood).
3. **Exact Equivariance of Physical Moves:**
   Let $m$ be any physical move in $s$ and $\tilde{m} = \mathcal{M}_{\text{move}}(m)$ be its vertically mirrored counterpart in $\mathcal{M}(s)$. By definition of $y_{\text{can}}(y, c_{\text{stm}})$, the canonical source square, destination square, and displacement $(\Delta x, \Delta y_{\text{can}})$ of $\tilde{m}$ under $1 - c_{\text{stm}}$ are identical to those of $m$ under $c_{\text{stm}}$, so $\gamma(\tilde{m}, 1 - c_{\text{stm}}) = \gamma(m, c_{\text{stm}}) = e$. Because $\Phi(\mathcal{C}(\mathcal{M}(s))) = \Phi(\mathcal{C}(s))$ bit-for-bit in `float32`, every intermediate token embedding $\mathbf{z}_i$ and edge output $f_\theta(\mathcal{M}(s))_e = f_\theta(s)_e$ is identical. $\blacksquare$

* **Empirical Verification (`SG2-A`):** Verified across 100 random 5x5 and 8x8 board states on `apple-m4-workstation` M4 Pro `mps` with maximum absolute deviation $\max_{s, e} |Q(s, e) - Q(\mathcal{M}(s), \tilde{e})| = \mathbf{0.000000}$ ($\le 10^{-5}$ target).

---

## 3. Theorem 3: Absorbing-State D3PM Convergence & 100% Legal Trajectory Support (`ALT-002`)

### 3.1 Formal Statement
Let $V = E \cup \{\texttt{[MASK]}\}$ be the augmented discrete action vocabulary of cardinality $|V| = |E| + 1$ ($456$ for $5 \times 5$; $1{,}859$ for $8 \times 8$), where index $m_{\text{id}} = |E|$ denotes the absorbing state $\texttt{[MASK]}$. For a horizon of $H_{\text{ply}} = 6$ plies, let $\mathbf{x}_0 = (e_0, \dots, e_5) \in E^6$ be a clean trajectory.

At each discrete diffusion step $t \in \{1, \dots, T\}$, the forward Markov transition matrix $Q_t \in [0, 1]^{|V| \times |V|}$ implemented in [`AbsorbingMaskSchedule`](../src/topo_chess/diffusion/d3pm_planner.py#L55-L100) is defined by:
$$Q_t = (1 - \beta_t) I_{|V|} + \beta_t \mathbf{1}_{|V|} \vec{u}_{m_{\text{id}}}^\top, \quad \beta_t \in (0, 1),$$
where $\vec{u}_{m_{\text{id}}}$ is the unit basis vector at index $m_{\text{id}}$.

**Theorem 3.**
1. *(Closed-Form Marginal & Stationary Convergence):* The $t$-step cumulative transition matrix $\bar{Q}_t = Q_1 Q_2 \cdots Q_t$ satisfies
   $$\bar{Q}_t = \bar{\alpha}_t I_{|V|} + (1 - \bar{\alpha}_t) \mathbf{1}_{|V|} \vec{u}_{m_{\text{id}}}^\top, \quad \text{where } \bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s).$$
   Every $Q_t$ and $\bar{Q}_t$ is strictly row-stochastic, $m_{\text{id}}$ is a recurrent absorbing state ($[Q_t]_{m_{\text{id}}, m_{\text{id}}} = 1$), and as $\bar{\alpha}_T \to 0$, $q(\mathbf{x}_T \mid \mathbf{x}_0) \to \delta_{(\texttt{[MASK]}, \dots, \texttt{[MASK]})}$ in total variation distance at rate $\mathcal{O}(\bar{\alpha}_T)$.
2. *(Exact Support on the Legal Trajectory Manifold):* Let $\mathcal{T}_{\text{legal}}(s_0) = \{(e_0, \dots, e_5) \in E^6 : \forall k \in \{0..5\},\, e_k \in \mathcal{L}(s_k) \text{ where } s_{k+1} = \mathrm{Step}(s_k, e_k)\}$ be the set of strictly legal 6-ply trajectories from $s_0$ (with post-terminal plies padded by the terminal mating/stalemating edge $e_{\text{last}}$ when $\mathcal{L}(s_k) = \varnothing$). Under the Two-Tier Legality Projection (`ALT-002`) in [`D3PMHorizonPlanner.sample_trajectories_batch`](../src/topo_chess/diffusion/d3pm_planner.py#L268-L403), the reverse sampling distribution $p_\theta^{\text{proj}}(\hat{\mathbf{x}}_0 \mid s_0)$ satisfies:
   $$\mathrm{supp}\big(p_\theta^{\text{proj}}(\cdot \mid s_0)\big) \subseteq \mathcal{T}_{\text{legal}}(s_0), \quad \text{and thus} \quad \mathbb{P}\big(\hat{\mathbf{x}}_0 \in \mathcal{T}_{\text{legal}}(s_0)\big) = 1.0000.$$

### 3.2 Proof
1. **Induction for $\bar{Q}_t$:**
   For $t = 1$, $\bar{Q}_1 = Q_1 = \bar{\alpha}_1 I + (1 - \bar{\alpha}_1) \mathbf{1} \vec{u}_{m_{\text{id}}}^\top$. Suppose the formula holds at $t-1$. Note that $(\mathbf{1} \vec{u}_{m_{\text{id}}}^\top) I = \mathbf{1} \vec{u}_{m_{\text{id}}}^\top$ and $(\mathbf{1} \vec{u}_{m_{\text{id}}}^\top)(\mathbf{1} \vec{u}_{m_{\text{id}}}^\top) = \mathbf{1}(\vec{u}_{m_{\text{id}}}^\top \mathbf{1})\vec{u}_{m_{\text{id}}}^\top = \mathbf{1} \vec{u}_{m_{\text{id}}}^\top$ since $\vec{u}_{m_{\text{id}}}^\top \mathbf{1} = 1$. Multiplying $\bar{Q}_{t-1} Q_t$:
   $$\bar{Q}_t = \big(\bar{\alpha}_{t-1} I + (1 - \bar{\alpha}_{t-1}) \mathbf{1}\vec{u}_{m_{\text{id}}}^\top\big)\big((1 - \beta_t)I + \beta_t \mathbf{1}\vec{u}_{m_{\text{id}}}^\top\big) = \bar{\alpha}_{t-1}(1 - \beta_t) I + \big(1 - \bar{\alpha}_{t-1}(1 - \beta_t)\big)\mathbf{1}\vec{u}_{m_{\text{id}}}^\top.$$
   Setting $\bar{\alpha}_t = \bar{\alpha}_{t-1}(1 - \beta_t)$ completes the induction. For each token $k \in \{0..5\}$, the total variation distance between $q(x_{T,k} \mid x_{0,k})$ and $\delta_{m_{\text{id}}}$ is $\bar{\alpha}_T$.
2. **Two-Tier Legality Projection Support:**
   * **Tier 1 (Steps $t > 1$):** [`predict_x0_logits`](../src/topo_chess/diffusion/d3pm_planner.py#L226-L227) sets $\ell_{b, k, m_{\text{id}}} = -10^9$, restricting intermediate token predictions strictly to $E$.
   * **Tier 2 (Final Projection Step $t = 0$):** Given initial board state $s_0$ with $\mathcal{L}(s_0) \ne \varnothing$, for $k = 0, \dots, 5$:
     * If $\mathcal{L}(s_k) \ne \varnothing$, logits for all $e \notin \mathcal{L}(s_k)$ are masked to $-\infty$, and $\hat{e}_k = \arg\max_{e \in \mathcal{L}(s_k)} \ell_{b, k, e} \in \mathcal{L}(s_k)$ is selected, advancing $s_{k+1} = \mathrm{Step}(s_k, \hat{e}_k)$.
     * If $\mathcal{L}(s_k) = \varnothing$ (game already terminated in checkmate or stalemate at ply $k' \le k$), $\hat{e}_k = \hat{e}_{k'-1}$ pads the sequence to fixed length $6$.
   By construction, at every non-terminal ply $k$, $\hat{e}_k \in \mathcal{L}(s_k)$, so $\hat{\mathbf{x}}_0 \in \mathcal{T}_{\text{legal}}(s_0)$ with probability $1.0000$. $\blacksquare$

* **Empirical Verification (`SG3-A` & `test_gate3`):** Verified across `10,000` test positions on `apple-m4-workstation` (`10,000 / 10,000` legal 6-ply trajectories, `100.0%`).

---

## 4. Proposition 4: Gauge-Invariant Fisher Covariance for Anchored BayesElo Ratings (`ALT-007`)

### 4.1 Formal Statement
In an $M$-player Bradley-Terry / BayesElo tournament with match counts $N_{ij}$ and win+half-draw scores $S_{ij}$, the negative log-likelihood is:
$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i < j} \Big[ N_{ij} \ln\big(1 + e^{\theta_i - \theta_j}\big) - S_{ij}(\theta_i - \theta_j) \Big].$$
Its Hessian $H_{\text{MLE}}(\boldsymbol{\theta}) = \nabla^2 \mathcal{L}(\boldsymbol{\theta})$ is a symmetric graph Laplacian with entries:
$$[H_{\text{MLE}}]_{ij} = -N_{ij} p_{ij}(1 - p_{ij}) \quad (i \ne j), \qquad [H_{\text{MLE}}]_{ii} = \sum_{j \ne i} N_{ij} p_{ij}(1 - p_{ij}),$$
where $p_{ij} = \sigma(\theta_i - \theta_j)$.

**Proposition 4.** *For any connected match graph and any positive gauge regularizer $\tau_0 > 0$, let $\Sigma(\tau_0) = (H_{\text{MLE}} + \tau_0 I_M)^{-1}$. For any player $i$ and fixed anchor player $a$, the relative variance*
$$V_{i \mid a} = (\vec{e}_i - \vec{e}_a)^\top \Sigma(\tau_0) (\vec{e}_i - \vec{e}_a) = \Sigma_{ii}(\tau_0) + \Sigma_{aa}(\tau_0) - 2\Sigma_{ia}(\tau_0)$$
*is completely independent of the nullspace regularizer $\tau_0$ in the limit $\tau_0 \to 0^+$ and equals $(\vec{e}_i - \vec{e}_a)^\top H_{\text{MLE}}^+ (\vec{e}_i - \vec{e}_a)$, where $H_{\text{MLE}}^+$ is the Moore-Penrose pseudoinverse.*

### 4.2 Proof
Since $H_{\text{MLE}}$ is the Laplacian of a connected graph, its eigenvalues satisfy $0 = \lambda_1 < \lambda_2 \le \dots \le \lambda_M$ with unit eigenvector $\vec{v}_1 = \frac{1}{\sqrt{M}} \mathbf{1}_M$ spanning $\ker(H_{\text{MLE}})$. By spectral decomposition:
$$\Sigma(\tau_0) = (H_{\text{MLE}} + \tau_0 I_M)^{-1} = \frac{1}{\tau_0} \vec{v}_1 \vec{v}_1^\top + \sum_{k=2}^{M} \frac{1}{\lambda_k + \tau_0} \vec{v}_k \vec{v}_k^\top.$$
Since $(\vec{e}_i - \vec{e}_a)^\top \vec{v}_1 = \frac{1}{\sqrt{M}}(1 - 1) = 0$, the divergent $\frac{1}{\tau_0} \vec{v}_1 \vec{v}_1^\top$ term vanishes identically:
$$(\vec{e}_i - \vec{e}_a)^\top \Sigma(\tau_0) (\vec{e}_i - \vec{e}_a) = \sum_{k=2}^{M} \frac{\big((\vec{e}_i - \vec{e}_a)^\top \vec{v}_k\big)^2}{\lambda_k + \tau_0} \xrightarrow{\tau_0 \to 0^+} (\vec{e}_i - \vec{e}_a)^\top H_{\text{MLE}}^+ (\vec{e}_i - \vec{e}_a). \quad \blacksquare$$
