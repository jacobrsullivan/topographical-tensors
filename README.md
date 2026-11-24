# Topographical Tensors: Geometric Move-Potential Chess & Certified 3D Digital Brick Assembly (`topographical-tensors`)

[![License: MIT](https://img.shields.io/badge/Code_License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0 / CC BY-SA 4.0](https://img.shields.io/badge/Catalog_Data-CC_BY_4.0_%2F_CC_BY--SA_4.0-teal.svg)](DATA_LICENSE.md)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Official PyTorch and optimization companion repository for **`topographical-tensors`** (`v5.4.1`), uniting two geometric discrete-planning domains under explicit local-frame invariants, sparse interaction graphs, discrete diffusion, and exact physical/algebraic certification:

1. **`topo_chess` (`src/topo_chess/`):** *Geometric Move-Potential Representations, $\mathbb{Z}_2$-Canonical Local-Frame Invariant Point Attention, and Single-Root Action-Only Discrete Diffusion (`D3PM`) for Searchless Chess* (`v5.0.0`).
2. **`topo_brick` (`src/topo_brick/`):** *Certified 3D Digital Brick Assembly via Exact Lattice/SE(3) Port Manifolds, Open-Catalog Sculptural Retrieval (`16,085` Parts), Finite Colored Inventories, Ribbed Hollow Shells, Compositional `.mpd` Subassemblies, and Contact-Wrench Equilibrium Certification* (`Gates 1–9D`).

---

## 1. Domain Overviews

### Domain I — `topo_brick`: Certified 3D Digital Brick Assembly (`src/topo_brick/`)

`topo_brick` separates deterministic geometric and mechanical invariants from learned spatial/combinatorial search via a four-operation **`TopoBrickKernel`** (`realize()`, `repair()`, `constrain()`, `certify()`), exposed to downstream consumers through a curated, lightweight `torch`-free **`topo_brick.api`** surface (`49` `SUPPORTED_CONSUMER_EXPORTS`, `RealizeRequestSpec`, `RealizeRouteProvenance`, `build_relying_party_obligations`, `verify_exported_mpd_roundtrip`, and `compare_realize_routes_on_target`):

* **Compiled `16,085`-Part Open Catalog, Calibrated `.dat` `SE(3)` Frames & `6` Connector Families:** Combines `213` curated `TIER_A` parts (all `213 / 213` calibrated with exact per-part official LDraw `.dat` `(R_0_ld, t_0_ld)` export/import frames; `0` frame errors across all `16,085` catalog IDs audited in [`docs/ldraw-dat-frame-audit-v5.2.1.json`](docs/ldraw-dat-frame-audit-v5.2.1.json) via `scripts/audit_ldraw_dat_frames.py`) with `16,072` compiled LDraw `2610` + LDCad Shadow parts (`99.52%` instance coverage and `99.99%` compatible port extraction across `26` human OMR `.mpd` sets; `0 / 16,072` compiled parts with dropped solid voxels via flush `(5, 5, 2) LDU` System grids), bore-carved compound `LDUBox` colliders (`0.0 LDU` wall-overlap tolerance), `6,831` `CoACD` convex polytope decompositions (`320` unique non-convex shells), and `6` connector families (`STUD_TUBE`, `TECHNIC_PIN_HOLE`, `TECHNIC_AXLE_HOLE`, `BAR_CLIP`, `BALL_SOCKET`, `HINGE_DETENT`). Supports strict `license_filter="cc_by_4_0_only"` (`0` `CC-BY-SA-4.0` parts emitted) for commercial-safe workflows.
* **Online Sculptural Mold Retrieval & Sub-Second Megastructures (`Gates 2, 7A, 7B, 8A, 8B`):**
  * **Blind Continuous Mesh-to-Brick (`n = 80` Strictly Unique Unseen Meshes, Single CPU Core):** Across `80` SHA-256-unique closed-manifold meshes (`4` external `BrickAnything` `.obj` + `76` fractional-stud procedural architectural meshes; `2` non-manifold `.obj` excluded from `82` candidates), coarse `S2_PairSeededLocalSearch` certifies `80 / 80` (`100%`) meshes with `80 / 80` arch openings clear in `47.8–71.6 ms` `p50` (`0.6994` original-mesh IoU, `86.25%` qualified `@0.60`, `3.85x–5.76x` faster than out-of-process `legolization==0.5.0` at `62.5%–72.5%` cert / `0.6002` IoU). Online 29-D volumetric/surface-normal mold retrieval (`Gate 8A`, `Top-48` over `1,499` deduplicated pose signatures across `426` active buildable System molds; `16,085 -> 16,072 -> 7,178 -> 426 / 1,499` funnel) actively places `44` distinct sloped, curved, wedge, tile, panel, SNOT, and arch molds across all `10 / 10` sculptural classes, reaching **`0.8542` overall mean / `0.8769` overall median / `0.8807` `Bands 2–4` mean (`0.9133` `Band 4`) `5×5×6 LDU` original-mesh IoU** (`80 / 80` certified & independently verified in `~291 ms` `p50`, `3.6 ms` retrieval `p50`), with perceptual CIELAB `CIEDE2000` palette quantization (`ΔE_00 = 5.72` mean) and `"smooth_studless"` tile finishing (`81.94%` exposed-stud reduction, `587 -> 106`).
  * **Megastructure Scaling, Tabletop Cantilevers & Ribbed Hollow Shells (`Gates 7B & 9B`):** `HierarchicalMacroBlockPlanner` certifies `36 / 36` (`100%`) multi-arch/cantilever megastructures (`263–952` pieces) in **`265.5–324.1 ms` `p50`** on 1 CPU core (and cantilevered tabletop biplane `28×26×9` in **`2.88 s`** / open-catalog `1,061`-part `28×26×24` airplane in **`29.98 s`** on M4 Pro under `table_friction_only` while preserving interplane wing windows, with shared `max_wall_clock_s` deadlines across both `HiGHS` and `scipy.optimize.linprog` returning `KernelResultStatus.BUDGET_EXCEEDED` within `<= 250 ms` grace), while `HollowShellPolicy(enabled=True)` (`Gate 9B`, `n = 30`, `16×16×10` to `24×24×16` studs under `FoundationPolicy.TABLE_FRICTION_ONLY`) cuts brick count by **`50.82%`** (`482.8 -> 235.4`) and plastic volume by **`54.65%`** (`0` temporary supports) with **`0.0001` mean `5×5×6 LDU` exterior-crust IoU difference** vs. solid fill.
  * **Color-Coupled Finite Inventories, Full `322`-Color `LDConfig.ldr` Palette, View Priority & Compositional `.mpd` Scenes (`Gates 9A, 9C, 9D`):** Embeds all `322` official `LDConfig.ldr` `!COLOUR` records (`ldraw_palette.py`, including `57` `Trans_Orange` `#F08F1C`, `ALPHA 128`, finish-aware transparent/pearlescent/metallic substitution penalties, and `0` gray fallback misses) and supports exact `(part_id, color_id)` finite bins (`ColoredInventorySpec`, e.g., official Rebrickable Set `10698-1` manifest `SET_10698_COLORED_INVENTORY` across `790` non-spare bricks / `108` molds / `33` LDraw colors) with `O(1)` inventory-masked `uint64` mold retrieval, parity-aware zero-`1×1` sparse-kit MILP tiling + `InventoryShortfallReport`, and two-zone core/skin min-cost chromatic network flow (`60 / 60` certified & verified in `~89.8–127.3 ms` `p50`, cutting visible skin `ΔE_00` by **`60.10%`**, `39.57 -> 15.79`); DDA camera-ray `ViewPrioritySpec` (`+0.0741` 2D silhouette IoU lift and `-11.14 ΔE_00` under `30%` color scarcity); and hierarchical rotated/nested multi-submodel `.mpd` `PLACE_SUBASSEMBLY` synthesis + `8`-role `LANDMARK_DETAIL_INDEX` (`110` mold records, upright `+Z` `BAR_POLE_ANTENNA` orientation & deterministic XY-centroid stud tie-breaking; `25 / 25` recursive `subtrace` & exact `SE(3)` round-trip).
  * **Constrained Autoregressive LLM Decoding & Deterministic SDK Adapter Tooling (`Gate 5`):** Multi-stream `BrickGrammarLogitsProcessor` over `AvaLovelace/BrickGPT`'s `128,256`-token `Llama-3` BPE vocabulary enforces `100%` action mask validity (`~5.0 ms/step`), and a matched-kit paired deterministic SDK adapter study (`EXTENDED_10PART_RECT_KIT`, `0` LLM calls) lifts `bricklayer-sdk==0.1.0` from `0 / 50` standalone to `47 / 50` (`94.0%`) certified builds while cutting emitted brick operations from `13.06` to `5.40`.
* **Planner-Free Independent Certificate Verification (`Gates 3 & 9A–9D`, Schema `v1.1.1`):** Standalone [`independent_verifier.py`](src/topo_brick/evaluation/independent_verifier.py) (`0` `topo_brick.kernel.*` or `static_equilibrium` imports; backward-compatible with `v1.1.0` and `v1.0.0`) enforces caller `VerificationObligations` (including explicit `ground_plane_z_ldu` / `expected_ground_plane_z_ldu` nonzero ground-plane datums), `MechanicsVerificationProfile`, and `VerifierResourceLimits` unconditionally (`verification_scope="assembly_only"` or `"assembly_and_target"`, reporting `target_digest_verified` and `target_overlap_verified` separately), verifies raw disk-byte SHA-256 digests of all bundled catalog/hull/index/inventory archives against pinned release provenance, replays `PLACE_PART` and rotated/nested `PLACE_SUBASSEMBLY` `subtrace`s with intermediate collision checks, checks continuous half-ray (`t in (0, +inf)`) 6-axis insertion corridors, and solves standalone `cubic_ldu` and `continuous_se3` 6-DoF `linprog` mechanics (`342 / 342` gate certificates replayed; `100%` mutant/tamper rejection).
* **Explicit Scope Boundaries:** Physical hardware validation (`Gate 1`, `8 / 8` CMU hardware benchmarks + `93.3%` zero-FP holdout accuracy across `120` unseen motif builds) applies strictly to `STUD_TUBE` interfaces; `PhysicalValidationCert` remains `UNKNOWN` unless an `ExternalPhysicalObservation` is attached (`0 / 120` in synthetic `Gate 4`, with `20 / 120` `Calibrated Sub-Stud Model Agreement`), monotone insertion checks 6-axis single-translation half-rays (`t in (0, +inf)`), and VLM prompt-aesthetic synthesis remains out of scope (caller-supplied mesh RGB/style hints map deterministically).

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
├── SECURITY.md                                         # Security policy, trust boundaries, and disclosure contact
├── LICENSE                                             # MIT License (code) + trademark & data-license pointer
├── DATA_LICENSE.md                                     # Per-file licensing for bundled catalog & benchmark data
├── THIRD_PARTY_NOTICES.md                              # Upstream LDraw, LDCad Shadow, CMU, and comparator notices
├── CITATION.cff                                        # GitHub citation metadata
├── pyproject.toml                                      # Package definition, dependencies, and wheel package-data
├── requirements/                                       # SHA-256 hashed Python 3.13 build & runtime lockfiles
├── Dockerfile                                          # Digest-pinned, non-root clean-room reproduction container
├── .github/workflows/
│   ├── ci.yml                                          # SHA-pinned PR/push CI (ruff + wheel + security & gate suites)
│   └── full-repro.yml                                  # Full 140-test Gate 1–9D, Phase 12–16.1 reproduction workflow
├── src/
│   ├── topo_brick/                                     # Certified 3D Digital Brick Assembly domain package
│   │   ├── interface/                                  # LDU/SE(3) frames, 16,085-part catalog, 322-color LDConfig, bins & 29-D retrieval
│   │   ├── representation/                             # 24-ch frontier port tensors & sparse tiled target fields
│   │   ├── models/                                     # SparsePortIPAEncoder, CandidateGraphDenoiser & outcome heads
│   │   ├── kernel/                                     # TopoBrickKernel, S1/S2, sculpted/hollow/colored/view/MPD realizers & LLM mask
│   │   ├── training/                                   # LDraw .ldr/.mpd I/O, course decomposition & value distillation
│   │   └── evaluation/                                 # 6-DoF HiGHS/SciPy LP validator, Schema v1.1.1 verifier & Gates 1–9D
│   └── topo_chess/                                     # Searchless Chess domain package (v5.0.0)
│       ├── domains/                                    # 8x8 chess, rectangular boards & 2D grid routing
│       ├── encoding/                                   # Bijective 44-ch Move-Potential encoder/decoder
│       ├── models/                                     # Unified 12-cell factorial encoders & local-frame IPA
│       ├── diffusion/                                  # Absorbing-state D3PM 6-ply horizon planner
│       ├── data/                                       # Zobrist-deduplicated Lichess & Stockfish PV pipelines
│       └── evaluation/                                 # 10k Lichess puzzles, 400-ply UCI gauntlets & joint-cov BayesElo
├── docs/                                               # Architecture specs, verification ledgers, assessments/, plans/, research/
│   ├── assessments/                                    # Capability & benchmark audits
│   ├── plans/                                          # Overall project roadmap & gate scorecards
│   └── research/                                       # Literature reviews, mathematical proofs, datasheets & synopses
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
* **PyTorch:** `2.10+` (`cpu`, Apple Silicon `mps`, or NVIDIA `cuda`; `torch>=2.10.0` enforced for restricted `weights_only=True` checkpoint safety)
* **Stockfish UCI Engine *(optional, required only for live `topo_chess` UCI gauntlets)*:** `brew install stockfish` (macOS) or `apt-get install stockfish` (Ubuntu)

### Setup
```bash
git clone https://github.com/jacobrsullivan/topographical-tensors.git
cd topographical-tensors

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Full dev + neural + chess suite (use `pip install -e .` for lightweight torch-free topo_brick core)
pip install -e ".[dev]"

# Verify hardware backend, HiGHS/Transformers, compiled 16,085-part brick catalog, and Stockfish
python scripts/check_env.py

# Run the <1s topo_brick sculpted mesh realization + independent certificate replay smoke test
python scripts/run_topo_brick_quickstart.py

# Run fast unit, contract, security hardening, and upstream TB-FIX / Phase 15–16.1 audit tests
pytest -v tests/test_p13_security_hardening_audit.py tests/test_topo_brick/test_p14_tb_fix_22nov_audit.py tests/test_topo_brick/test_p15_architecture_and_policy_audit.py tests/test_topo_brick/test_p16_v2_assessment_remediation_audit.py tests/test_topo_brick/test_p16_1_closure_audit.py tests/test_topo_brick/test_wp_l1_interface.py tests/test_topo_brick/test_revised_construction_kernel.py tests/test_topo_brick/test_p12_contract_and_falsification_audit.py

# Run the complete 140-test topo_brick Gate 1–9D + Phase 12–16.1 suite and topo_chess suite
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 pytest -v tests/
```

*(Optional extras: `pip install -e .` installs the lightweight `numpy`/`scipy`/`highspy` `topo_brick` core without `torch`; `pip install -e ".[neural]"` adds `torch` + `transformers`; `pip install -e ".[chess]"` adds `torch` + `chess`; `pip install -e ".[compiler]"` adds `coacd` + `pyarrow`.)*

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

### `topo_brick` Architecture, Data & Verification (`Gates 1–9D`)
* **Overall Project Plan & Gate Scorecard (`Gates 1–9D`):** [`docs/plans/lego-project-overall-plan.md`](docs/plans/lego-project-overall-plan.md)
* **Phase 11 Five Extensions Specification (`v5.2.0`):** [`docs/five-extensions.md`](docs/five-extensions.md)
* **Capability Assessment & Benchmark Review (`130` Tests, `14` Gates):** [`docs/assessments/capability-assessment-review.md`](docs/assessments/capability-assessment-review.md)
* **System Architecture & Kernel Interfaces:** [`docs/system-architecture-and-interfaces.md`](docs/system-architecture-and-interfaces.md)
* **Model Architecture, Sculptural Retrieval & Solvers:** [`docs/model-architecture-and-component-design.md`](docs/model-architecture-and-component-design.md)
* **Data Architecture, Provenance Tiers & Collider Contracts:** [`docs/data-architecture-and-pipeline.md`](docs/data-architecture-and-pipeline.md)
* **Foundational Lattice & Port-Manifold Guide:** [`docs/topo-brick-expert-guide.md`](docs/topo-brick-expert-guide.md)
* **Literature Review (`2015–2026`):** [`docs/research/lego-literature-review.md`](docs/research/lego-literature-review.md)
* **Dual-Host Gate & Frame-Audit Ledgers:** [`docs/phase9-gate-verification-ledger.json`](docs/phase9-gate-verification-ledger.json) · [`docs/phase10-gate-verification-ledger.json`](docs/phase10-gate-verification-ledger.json) · [`docs/phase11-gate-verification-ledger.json`](docs/phase11-gate-verification-ledger.json) · [`docs/ldraw-dat-frame-audit-v5.2.1.json`](docs/ldraw-dat-frame-audit-v5.2.1.json)

### `topo_chess` Manuscript & Proofs (`v5.0.0`)
* **Main-Conference Manuscript:** [`paper/manuscript_iclr_neurips_icml.md`](paper/manuscript_iclr_neurips_icml.md) ([LaTeX](paper/manuscript_iclr_neurips_icml.tex))
* **Executive Research Synopsis (`v5.0.0`):** [`docs/research/12-expert-review-synopsis.md`](docs/research/12-expert-review-synopsis.md)
* **Formal Mathematical Proofs (Theorems 1–3 & Proposition 4):** [`docs/research/06-mathematical-proofs-and-theoretical-guarantees.md`](docs/research/06-mathematical-proofs-and-theoretical-guarantees.md)

---

## 6. Licensing & Trademarks

* **Source Code:** Licensed under the **MIT License** ([`LICENSE`](LICENSE)); see [`SECURITY.md`](SECURITY.md) for trust boundaries and vulnerability reporting.
* **Bundled Catalogs & Benchmark Datasets:** Governed by their respective **`CC-BY-4.0`**, **`CC-BY-SA-4.0`**, **`CCAL v2.0`**, and **`MIT`** terms as detailed in [`DATA_LICENSE.md`](DATA_LICENSE.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
* **Trademark Disclaimer:** LEGO® is a trademark of the LEGO Group of companies which does not sponsor, authorize, or endorse this software or research. LDraw™ is a trademark owned and licensed by the Estate of James Jessiman.

---

## 7. Citation

```bibtex
@software{sullivan2026topographical,
  title   = {Topographical Tensors: Geometric Move-Potential Chess and Certified 3D Digital Brick Assembly},
  author  = {Sullivan, Jacob},
  version = {5.4.1},
  year    = {2026},
  url     = {https://github.com/jacobrsullivan/topographical-tensors}
}
```
