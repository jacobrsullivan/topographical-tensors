# Geometric Move-Potential Representations and Action-Only Discrete Diffusion for Searchless Chess (`topographical-tensors`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Official PyTorch implementation and manuscript companion repository for **"Geometric Move-Potential Representations and Action-Only Discrete Diffusion for Searchless Chess"** (`v5.0.0`).

---

## 1. Overview

`topographical-tensors` (`topo_chess`) investigates whether explicit board-geometry inductive biases—both in the input state representation and in the spatial attention kernel—combined with single-root action-only discrete diffusion can improve move-ranking fidelity, tactical puzzle solving, and searchless gameplay under a fixed single-workstation data budget (`1,269,820` Zobrist-deduplicated Lichess positions and `250,000` 6-ply Stockfish principal-variation trajectories):

1. **44-Channel Move-Potential Representation (`topo_44`):** Augments the standard 20-channel discrete board encoding (`flat_20`, channels `24:44`) with 24 deterministic, rule-derived continuous scalar and vector fields (channels `0:24`) capturing first-blocker-attenuated sliding ray potentials, net control flux, spatial flux gradients, collinear pin tension, second-blocker X-ray transmission, and harmonic king tropism (`0.067 ms/state` on GPU).
2. **$\mathbb{Z}_2$-Canonical Local-Frame Invariant Point Attention (`local_frame_ipa`) & Graph Sparsity:** Projects spatial query/key points into canonical square-local coordinate frames restricted to the sparse directed graph of legal chess move trajectories (`1,520` `queen_rays` or `1,856` `queen_plus_knight` edges out of `4,096` dense pairs, cutting pairwise attention FLOPs by **`69.8%`**) with exact $\mathbb{Z}_2$ vertical-mirror + color-inversion equivariance (`0.000000` error in `float32`).
3. **Single-Root Action-Only Discrete State-Space Diffusion (`D3PMHorizonPlanner`):** Iteratively denoises 6-ply move token sequences $a_{0:5} \in \{0, \dots, 1858\}^6$ from an absorbing `[MASK]` state conditioned on a single cached root board encoding $z(s_0) \in \mathbb{R}^{64 \times 256}$ (`0.905 ms/state` at $T=20$, **`17.4x` faster** than width-4 re-encoded beam search), guided by an intrinsic multi-ply legality head (`ALT-009`).

---

## 2. Repository Structure

```text
topographical-tensors/
├── README.md                                           # Project overview, installation, quickstart, and citation
├── LICENSE                                             # MIT License
├── CITATION.cff                                        # GitHub citation metadata
├── pyproject.toml                                      # Python package definition, dependencies, and test/lint config
├── Dockerfile                                          # Deterministic clean-room reproduction container
├── .env.example                                        # Environment configuration template (TOPO_DATA_DIR, STOCKFISH_PATH)
├── .github/workflows/ci.yml                            # Automated GitHub Actions CI (ruff + pytest)
├── paper/                                              # Main-conference submission manuscript (ICLR / NeurIPS / ICML)
│   ├── manuscript_iclr_neurips_icml.md                 # Full Markdown paper + Appendices A–D
│   └── manuscript_iclr_neurips_icml.tex                # LaTeX source for ICLR / NeurIPS / ICML compilation
├── docs/                                               # Mathematical proofs, datasheets, and research synopsis
│   ├── 01-mathematical-and-engineering-tensions.md     # Architectural design notes (ALT-001..009)
│   ├── 06-mathematical-proofs-and-theoretical-guarantees.md # Formal proofs (Theorems 1–3 & Proposition 4)
│   ├── 07-hyperparameters-datasheet-and-compute-audit.md    # Hyperparameters, compute audit & dataset datasheet
│   ├── 08-neurips-checklist-and-ste100-editorial-audit.md   # Reproducibility checklist
│   ├── 12-expert-review-synopsis.md                         # Self-contained executive research synopsis (v5.0.0)
│   └── 17-main-conference-manuscript.md                     # Markdown mirror of the main-conference manuscript
├── configs/                                            # Work Package & factorial ablation YAML configs
│   ├── wp1_data_pipeline.yaml
│   ├── wp2_ipa_encoder.yaml
│   ├── wp3_d3pm_planner.yaml
│   └── wp4_ablation_matrix.yaml
├── evaluations/                                        # Exported Markdown benchmark and tournament tables
├── logs/                                               # Exported JSON benchmark and factorial telemetry
├── src/
│   └── topo_chess/                                     # Core library package
│       ├── domains/                                    # 5x5 Gardner (455 edges), 8x8 Standard (1,858 edges) & 2D Grid Routing
│       ├── data/                                       # Zobrist deduplication, Stockfish UCI distillation & Lichess shards
│       ├── encoding/                                   # CPU & batched GPU Move-Potential Tensor encoder + lossless decoder
│       ├── models/                                     # Local-Frame IPA, Shaw 2D, Czech ConvPlanes28, 12-Cell Factorial & Controls
│       ├── diffusion/                                  # Absorbing [MASK] D3PM scheduler, ALT-009 legality head & AR baseline
│       └── evaluation/                                 # Kendall tau, 10k Lichess puzzle suite, UCI gauntlets & joint-cov BayesElo
├── tests/                                              # Automated pytest verification suite
│   ├── test_gate1_encoding_and_hygiene.py
│   ├── test_gate2_ipa_equivariance.py
│   ├── test_gate3_d3pm_legality.py
│   ├── test_gate4_ablation_and_elo.py
│   ├── test_gate5_clean_room_repro.py
│   ├── test_phase_b_8x8_pipeline.py
│   ├── test_phase_c_full_scale.py
│   ├── test_phase_d_audit_compliance.py
│   └── test_phase_e_round2_compliance.py
└── scripts/                                            # CLI entrypoints & benchmark runners
    ├── check_env.py
    ├── run_wp1_wp2_wp3_verification.py
    ├── run_wp4_ablation_and_benchmarks.py
    ├── run_wp5_clean_room_repro.py
    ├── run_phase_b_8x8_pipeline.py
    ├── run_phase_c_full_scale.py
    ├── run_phase_d_v4_audit.py
    └── run_phase_e_v5_audit.py
```

---

## 3. Installation & Quickstart

### Prerequisites
* **Python:** `3.11+`
* **PyTorch:** `2.4+` (supports Apple Silicon `mps`, NVIDIA `cuda`, or `cpu`)
* **Stockfish UCI Engine:** Stockfish 16–19 installed on `PATH` (e.g., `brew install stockfish` on macOS or `apt-get install stockfish` on Ubuntu)

### Setup
```bash
git clone https://github.com/jacobrsullivan/topographical-tensors.git
cd topographical-tensors

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Verify environment, hardware backend, and Stockfish UCI binary
python scripts/check_env.py

# Run the automated test suite
pytest -v
```

### Minimal Python Usage
```python
import torch
from topo_chess.domains.boards import Standard8x8State
from topo_chess.encoding.topographical_encoder import TopographicalDecoder, TopographicalEncoder
from topo_chess.models.phase_d_controls import UnifiedSpatialEncoder

# 1. Encode a standard 8x8 chess position into its 44-channel Move-Potential Tensor
state = Standard8x8State.from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
encoder_44 = TopographicalEncoder(height=8, width=8)
tensor_44 = encoder_44.encode_canonical(state)  # shape: (44, 8, 8)

# Verify exact algebraic bijectivity via slices 24:44
decoded = TopographicalDecoder(height=8, width=8).decode_canonical(tensor_44, original_stm=state.side_to_move)
assert decoded.to_fen() == state.to_fen()

# 2. Run Cell_12 (topo_44 + Local-Frame IPA + Queen+Knight 1,856-edge sparse mask)
model = UnifiedSpatialEncoder(
    cell_name="Cell_12_Topo44_IPA_QueenKnight",
    input_mode="topo_44",
    attention_kernel="local_frame_ipa",
    connectivity_mask="queen_plus_knight",
    d_model=256,
    num_layers=8,
    n_heads=8,
    d_ff=768,
)
x_batch = torch.from_numpy(tensor_44).unsqueeze(0)  # (1, 44, 8, 8)
out = model(x_batch)
print("Action-value logits shape:", out.q_logits.shape)  # (1, 1858)
```

---

## 4. Paper & Documentation

* **Main-Conference Manuscript (Markdown):** [`paper/manuscript_iclr_neurips_icml.md`](paper/manuscript_iclr_neurips_icml.md)
* **Main-Conference Manuscript (LaTeX):** [`paper/manuscript_iclr_neurips_icml.tex`](paper/manuscript_iclr_neurips_icml.tex)
* **Executive Research Synopsis (`v5.0.0`):** [`docs/12-expert-review-synopsis.md`](docs/12-expert-review-synopsis.md)
* **Formal Mathematical Proofs (Theorems 1–3 & Proposition 4):** [`docs/06-mathematical-proofs-and-theoretical-guarantees.md`](docs/06-mathematical-proofs-and-theoretical-guarantees.md)

---

## 5. Citation

```bibtex
@article{sullivan2026topographical,
  title   = {Geometric Move-Potential Representations and Action-Only Discrete Diffusion for Searchless Chess},
  author  = {Sullivan, Jacob},
  year    = {2026},
  url     = {https://github.com/jacobrsullivan/topographical-tensors}
}
```
