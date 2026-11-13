# Topographical Tensors: Geometric Move-Potential Chess & Certified 3D Digital Brick Assembly (`topographical-tensors`)

[![License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0 / CC BY-SA 4.0](https://img.shields.io/badge/Catalog_Data-CC_BY_4.0_%2F_CC_BY--SA_4.0-teal.svg)](DATA_LICENSE.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Official PyTorch and optimization companion repository for **`topographical-tensors`** (`v5.1.0`), uniting two geometric discrete-planning domains under explicit local-frame invariants, sparse interaction graphs, discrete diffusion, and exact physical/algebraic certification:

1. **`topo_chess` (`src/topo_chess/`):** *Geometric Move-Potential Representations, $\mathbb{Z}_2$-Canonical Local-Frame Invariant Point Attention, and Single-Root Action-Only Discrete Diffusion (`D3PM`) for Searchless Chess* (`v5.0.0`).
2. **`topo_brick` (`src/topo_brick/`):** *Certified 3D Digital Brick Assembly via Exact Lattice/SE(3) Port Manifolds, Open-Catalog Sculptural Retrieval (`16,085` Parts), Hierarchical Macro-Blocks, and Contact-Wrench Equilibrium Certification* (`Gates 1–8B`).

---

## 1. Domain Overviews

### Domain I — `topo_brick`: Certified 3D Digital Brick Assembly (`src/topo_brick/`)

`topo_brick` separates deterministic geometric and mechanical invariants from learned spatial/combinatorial search via a four-operation **`TopoBrickKernel`** (`realize()`, `repair()`, `constrain()`, `certify()`):

* **Compiled `16,085`-Part Open Catalog & `6` Connector Families:** Combines `213` curated `TIER_A` parts with `16,072` compiled LDraw `2610` + LDCad Shadow parts (`99.52%` instance coverage across `26` human OMR `.mpd` sets), bore-carved compound `LDUBox` colliders (`0.0 LDU` wall-overlap tolerance), `3,273` `CoACD` convex polytope decompositions, and `6` connector families (`STUD_TUBE`, `TECHNIC_PIN_HOLE`, `TECHNIC_AXLE_HOLE`, `BAR_CLIP`, `BALL_SOCKET`, `HINGE_DETENT`). Supports strict `license_filter="cc_by_4_0_only"` (`0` `CC-BY-SA-4.0` parts emitted) for commercial-safe workflows.
* **Online Sculptural Mold Retrieval & Sub-Second Megastructures (`Gates 2, 7A, 7B, 8A, 8B`):**
  * **Blind Continuous Mesh-to-Brick (`n = 80` Unseen Meshes, Single CPU Core):** Coarse `S2_PairSeededLocalSearch` certifies `80 / 80` (`100%`) meshes with `80 / 80` arch openings clear in `52.9–66.1 ms` `p50` (`0.7531` original-mesh IoU, `4.95x–5.22x` faster than out-of-process `legolization==0.5.0` at `61.25%` cert / `0.5998` IoU). Online 29-D volumetric/surface-normal mold retrieval (`Gate 8A`, `Top-48` over `849` mold signatures) actively places `46–52` distinct sloped, curved, wedge, tile, and arch molds across all `9 / 9` sculptural classes, reaching **`0.9064` overall mean / `0.9295` overall median / `0.9349` `Bands 2–4` mean `5×5×6 LDU` original-mesh IoU** (`80 / 80` certified in `133.8–205.1 ms` `p50`), with perceptual CIELAB `CIEDE2000` palette quantization (`ΔE_00 = 5.54` mean) and `"smooth_studless"` tile finishing (`87.52%` exposed-stud reduction, `569 -> 71`).
  * **Megastructure Scaling (`Gate 7B`, `n = 36` Meshes of `263–952` Pieces):** `HierarchicalMacroBlockPlanner` certifies `36 / 36` (`100%`) multi-arch/cantilever megastructures in **`223–365 ms` `p50` (`612–943 ms` `p90`)** on 1 CPU core (`24.8x` faster than flat global MILP).
  * **Constrained Autoregressive LLM Decoding & Agent Tooling (`Gate 5`):** Live `BrickGrammarLogitsProcessor` over `AvaLovelace/BrickGPT`'s `128,256`-token `Llama-3` BPE vocabulary achieves `100%` action validity (`15.2–16.6 ms/step` `p50`), and paired agent A/B evaluation lifts `bricklayer-sdk==0.1.0` from `33.3%` to `100.0%` (`60 / 60`) certified builds.
* **Planner-Free Independent Certificate Verification (`Gate 3`):** Standalone [`independent_verifier.py`](src/topo_brick/evaluation/independent_verifier.py) replays SHA-256 catalog digests, `2 LDU` (`0.8 mm`) CSG cavity-carved collision grids, monotone insertion corridors, and primal 6-DoF contact-wrench witnesses (`100%` mutant rejection across `32 / 32` falsification cases; `0` planner imports).
* **Explicit Scope Boundaries:** Physical force/torque calibration (`Gate 1`, `88.5%` overall accuracy across `200` experimental/reference assemblies) applies strictly to `STUD_TUBE` interfaces; non-stud connectors carry kinematic/topological wrench certificates rather than empirical destructive load bounds, and VLM prompt-aesthetic synthesis remains out of scope (Option 3B maps caller-supplied mesh RGB/style hints deterministically).

### Domain II — `topo_chess`: Geometric Move-Potential & Discrete Diffusion for Searchless Chess (`src/topo_chess/`)

`topo_chess` (`v5.0.0`) investigates explicit board-geometry inductive biases and single-root action-only discrete diffusion under a fixed single-workstation data budget (`1,269,820` Zobrist-deduplicated Lichess positions and `250,000` 6-ply Stockfish principal-variation trajectories):

1. **44-Channel Move-Potential Representation (`topo_44`):** Augments the 20-channel discrete board encoding (`flat_20`) with 24 rule-derived continuous ray-potential, control-flux, pin-tension, X-ray, and harmonic king-tropism fields (`0.067 ms/state` on GPU).
2. **$\mathbb{Z}_2$-Canonical Local-Frame Invariant Point Attention (`local_frame_ipa`):** Restricts attention to sparse legal chess move trajectories (`1,520` `queen_rays` or `1,856` `queen_plus_knight` edges out of `4,096` dense pairs, cutting pairwise attention FLOPs by `69.8%`) with exact $\mathbb{Z}_2$ color-inversion equivariance.
3. **Single-Root Action-Only Discrete Diffusion (`D3PMHorizonPlanner`):** Denoises 6-ply move sequences $a_{0:5} \in \{0, \dots, 1858\}^6$ from a single cached root board encoding (`0.905 ms/state` at $T=20$, `17.4x` faster than width-4 re-encoded beam search).

---

## 2. Repository Structure

```text
topographical-tensors/
├── README.md                                           # Overview, setup, dual-domain quickstart, and citation
├── LICENSE                                             # MIT License (code) + trademark & data-license pointer
├── DATA_LICENSE.md                                     # Per-file licensing for bundled catalog & benchmark data
├── THIRD_PARTY_NOTICES.md                              # Upstream LDraw, LDCad Shadow, CMU, and comparator notices
├── CITATION.cff                                        # GitHub citation metadata
├── pyproject.toml                                      # Package definition, dependencies, and wheel package-data
├── Dockerfile                                          # Deterministic clean-room reproduction container
├── .github/workflows/
│   ├── ci.yml                                          # Fast PR/push CI (ruff + wheel audit + chess & brick gates)
│   └── full-repro.yml                                  # Full 64-test Gate 1–8B & Phase E reproduction workflow
├── src/
│   ├── topo_brick/                                     # Certified 3D Digital Brick Assembly domain package
│   │   ├── interface/                                  # LDU/SE(3) frames, 16,085-part catalog, CoACD & 29-D retrieval
│   │   ├── representation/                             # 24-ch frontier port tensors & sparse tiled target fields
│   │   ├── models/                                     # SparsePortIPAEncoder, CandidateGraphDenoiser & outcome heads
│   │   ├── kernel/                                     # TopoBrickKernel, S1/S2 solvers, sculpted skinning, macro-blocks & LLM mask
│   │   ├── training/                                   # LDraw .ldr/.mpd I/O, course decomposition & value distillation
│   │   └── evaluation/                                 # 6-DoF HiGHS LP validator, independent verifier & Gates 1–8B
│   └── topo_chess/                                     # Searchless Chess domain package (v5.0.0)
│       ├── domains/                                    # 8x8 chess, rectangular boards & 2D grid routing
│       ├── encoding/                                   # Bijective 44-ch Move-Potential encoder/decoder
│       ├── models/                                     # Unified 12-cell factorial encoders & local-frame IPA
│       ├── diffusion/                                  # Absorbing-state D3PM 6-ply horizon planner
│       ├── data/                                       # Zobrist-deduplicated Lichess & Stockfish PV pipelines
│       └── evaluation/                                 # 10k Lichess puzzles, 400-ply UCI gauntlets & joint-cov BayesElo
├── docs/                                               # Technical architecture specs, proofs, and verification ledgers
├── paper/                                              # Searchless Chess manuscript (Markdown & LaTeX)
├── configs/                                            # YAML configurations for topo_chess factorial studies
├── evaluations/ & logs/                                # Exported tables and JSON audit ledgers
├── scripts/                                            # Runnable environment checks, quickstarts, and benchmark CLIs
└── tests/                                              # Automated pytest suites (tests/test_*.py & tests/test_topo_brick/)
```

---

## 3. Installation & Verification

### Prerequisites
* **Python:** `3.11+` (`3.11`–`3.13` tested)
* **PyTorch:** `2.4+` (`cpu`, Apple Silicon `mps`, or NVIDIA `cuda`)
* **Stockfish UCI Engine *(optional, required only for live `topo_chess` UCI gauntlets)*:** `brew install stockfish` (macOS) or `apt-get install stockfish` (Ubuntu)

### Setup
```bash
git clone https://github.com/jacobrsullivan/topographical-tensors.git
cd topographical-tensors

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Verify hardware backend, HiGHS/Transformers, compiled 16,085-part brick catalog, and Stockfish
python scripts/check_env.py

# Run the <1s topo_brick sculpted mesh realization + independent certificate replay smoke test
python scripts/run_topo_brick_quickstart.py

# Run fast unit & kernel verification tests
pytest -v tests/test_topo_brick/test_wp_l1_interface.py tests/test_topo_brick/test_revised_construction_kernel.py

# Run the complete 64-test topo_brick Gate 1–8B suite + 9-file topo_chess suite
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -v tests/
```

*(Optional offline catalog re-compilation from raw LDraw `2610` + LDCad Shadow sources uses `pip install -e ".[compiler]"` for `coacd` and `pyarrow`.)*

---

## 4. Minimal Python Usage

### 4.1 `topo_brick`: Sculpted Mesh-to-Brick Realization & Independent Certificate Replay
```python
from topo_brick import (
    AdaptiveSurfaceTarget,
    TopoBrickKernel,
    export_certificate_artifact,
    load_frozen_80_unseen_mesh_corpus,
    verify_certificate_independently,
    voxelize_triangle_mesh_to_target_field,
)

# 1. Load an unseen continuous 3D triangle mesh with a required clear arch opening
corpus, _ = load_frozen_80_unseen_mesh_corpus()
mesh = next(m for m in corpus if len(m["required_openings_ldu"]) > 0)

coarse_target, _ = voxelize_triangle_mesh_to_target_field(
    mesh["verts"], mesh["faces"], max_xy_studs=mesh["max_xy_studs"], max_z_courses=mesh["max_z_courses"]
)
adaptive_target = AdaptiveSurfaceTarget.from_triangle_mesh(
    mesh["verts"], mesh["faces"], max_xy_studs=mesh["max_xy_studs"], max_z_courses=mesh["max_z_courses"]
)

# 2. Realize with online Top-32 sculptural mold retrieval (strictly CC-BY-4.0 parts + studless finish)
kernel = TopoBrickKernel()
result = kernel.realize(
    coarse_target,
    solver_id="S1_CoverAndRepair",
    part_retrieval="open_catalog_top_k",
    retrieval_top_k=32,
    license_filter="cc_by_4_0_only",
    semantic_style_hint="smooth_studless",
    surface_refinement="open_catalog_sculpted",
    adaptive_mesh_target=adaptive_target,
    support_policy="minimal_permanent_pillars",
    forbidden_support_boxes=tuple(mesh["required_openings_ldu"]),
)

# 3. Export LDraw .mpd and replay verification independently (zero planner imports)
mpd_text = result.assembly.to_ldraw_mpd(model_name="arch_shrine.ldr")
cert = export_certificate_artifact(
    result.assembly, result.certificates, validator=kernel.validator,
    target_field=coarse_target, support_policy="minimal_permanent_pillars",
)
audit = verify_certificate_independently(cert)
assert audit.verified
```

### 4.2 `topo_chess`: 44-Channel Move-Potential Encoding & Sparse Local-Frame IPA
```python
import torch
from topo_chess.domains.boards import Standard8x8State
from topo_chess.encoding.topographical_encoder import TopographicalDecoder, TopographicalEncoder
from topo_chess.models.phase_d_controls import UnifiedSpatialEncoder

state = Standard8x8State.from_fen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
tensor_44 = TopographicalEncoder(height=8, width=8).encode_canonical(state)  # (44, 8, 8)
assert TopographicalDecoder(height=8, width=8).decode_canonical(tensor_44, original_stm=state.side_to_move).to_fen() == state.to_fen()

model = UnifiedSpatialEncoder(
    cell_name="Cell_12_Topo44_IPA_QueenKnight",
    input_mode="topo_44",
    attention_kernel="local_frame_ipa",
    connectivity_mask="queen_plus_knight",
    d_model=256, num_layers=8, n_heads=8, d_ff=768,
)
logits = model(torch.from_numpy(tensor_44).unsqueeze(0)).q_logits  # (1, 1858)
```

---

## 5. Documentation Index

### `topo_brick` Architecture, Data & Verification (`Gates 1–8B`)
* **Overall Project Plan & Gate Scorecard:** [`docs/lego-project-overall-plan.md`](docs/lego-project-overall-plan.md)
* **Capability Assessment & Benchmark Review (`Gates 1–8B`):** [`docs/capability-assessment-review.md`](docs/capability-assessment-review.md)
* **System Architecture & Kernel Interfaces:** [`docs/system-architecture-and-interfaces.md`](docs/system-architecture-and-interfaces.md)
* **Model Architecture, Sculptural Retrieval & Solvers:** [`docs/model-architecture-and-component-design.md`](docs/model-architecture-and-component-design.md)
* **Data Architecture, Provenance Tiers & Collider Contracts:** [`docs/data-architecture-and-pipeline.md`](docs/data-architecture-and-pipeline.md)
* **Foundational Lattice & Port-Manifold Guide:** [`docs/topo-brick-expert-guide.md`](docs/topo-brick-expert-guide.md)
* **Literature Review (`2015–2026`):** [`docs/lego-literature-review.md`](docs/lego-literature-review.md)
* **Dual-Host Gate Ledgers:** [`docs/phase9-gate-verification-ledger.json`](docs/phase9-gate-verification-ledger.json) & [`docs/phase10-gate-verification-ledger.json`](docs/phase10-gate-verification-ledger.json)

### `topo_chess` Manuscript & Proofs (`v5.0.0`)
* **Main-Conference Manuscript:** [`paper/manuscript_iclr_neurips_icml.md`](paper/manuscript_iclr_neurips_icml.md) ([LaTeX](paper/manuscript_iclr_neurips_icml.tex))
* **Executive Research Synopsis (`v5.0.0`):** [`docs/12-expert-review-synopsis.md`](docs/12-expert-review-synopsis.md)
* **Formal Mathematical Proofs (Theorems 1–3 & Proposition 4):** [`docs/06-mathematical-proofs-and-theoretical-guarantees.md`](docs/06-mathematical-proofs-and-theoretical-guarantees.md)

---

## 6. Licensing & Trademarks

* **Source Code:** Licensed under the **MIT License** ([`LICENSE`](LICENSE)).
* **Bundled Catalogs & Benchmark Datasets:** Governed by their respective **`CC-BY-4.0`**, **`CC-BY-SA-4.0`**, **`CCAL v2.0`**, and **`MIT`** terms as detailed in [`DATA_LICENSE.md`](DATA_LICENSE.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
* **Trademark Disclaimer:** LEGO® is a trademark of the LEGO Group of companies which does not sponsor, authorize, or endorse this software or research. LDraw™ is a trademark owned and licensed by the Estate of James Jessiman.

---

## 7. Citation

```bibtex
@software{sullivan2026topographical,
  title   = {Topographical Tensors: Geometric Move-Potential Chess and Certified 3D Digital Brick Assembly},
  author  = {Sullivan, Jacob},
  version = {5.1.0},
  year    = {2026},
  url     = {https://github.com/jacobrsullivan/topographical-tensors}
}
```
