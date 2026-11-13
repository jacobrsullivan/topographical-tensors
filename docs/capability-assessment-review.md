# Topo-Brick Capability Assessment & Verified Test Review

**Verification Date:** 2026-11-13  
**Scope:** Complete live execution, standalone benchmark reconciliation, independent certificate replay, and code-level forensic audit of all **64 `pytest` tests across all 14 test modules** in `tests/test_topo_brick/`, covering Stages 1–6, Round 7 Deliverables 1–4, Phase 6 (`WS-1`–`WS-6`), Phase 7 (`R8-WS1`–`R8-WS6`), Phase 8 (`Round 9 Gates 1–5`), Phase 9 (`Gates 6, 7A, 7B`: Automated `16,085`-Part LDraw + LDCad Shadow + `CoACD` Catalog Compiler, Fast Adaptive Plate + Slope Surface Skinning, and Sub-Second Hierarchical Macro-Block Planning), and **Phase 10 (`Gates 8A & 8B`: Open-Catalog `uint64` `Top-K` Retrieval Over `6,825` Canonical System Molds, `10 LDU` Jumper + `24`-Pose SNOT + Curved/Steep/Inverted Slope + Wedge Sculpting to `0.9064` Mean / `0.9295` Median / `0.9349` `Bands 2–4` Original-Mesh IoU, Strict `CC-BY-4.0` Commercial Mode, and Option 3B `CIEDE2000` Chromatic & Studless-Tile Skinning)**.

> [!IMPORTANT]
> **Methodological & Reproducibility Commitment (`Gates 1–8B`):**
> 1. **True Single-Core & Resource Accounting:** All online benchmarks pin `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)`, and HiGHS `threads=1` (`cpu_to_wall_ratio = 0.9982–1.0000`), reporting **process `Peak RSS`** (`395.3–839.8 MB` Host B / `512.6–991.7 MB` Host A) separately from **`SparseTiledTargetField` active boundary tile memory (`2.4279 MiB/tile`)**.
> 2. **Standalone Benchmark Runs vs. `pytest` Regression Suite:** Full-scale multi-thousand-build benchmarks (such as the `2,400`-realization `S2B-600` standalone run taking `3,355.1 s` Host B / `4,013.1 s` Host A, out-of-process `hbmartin/legolization` runs, `docs/phase9-gate-verification-ledger.json`, and `docs/phase10-gate-verification-ledger.json`) are archived in versioned JSON ledgers; `pytest` verifies schema/CI invariants against those ledgers while executing live stratified slices and full `Gates 1–8B` checks (`64 / 64` passing).
> 3. **Planner-Free Certificate Replay & Dual Catalog/Index Hash Binding:** [`independent_verifier.py`](../src/topo_brick/evaluation/independent_verifier.py) (`--self-test`) binds both `COMPILED_LDRAW_CATALOG.archive_sha256` and `compiled_mold_retrieval_index.npz` SHA-256 directly into `compute_catalog_sha256()` (`436a96810107192001bccd3e104d006bf3d8b7330ad13e18390059c8af3b47b4`) with **zero planner or retrieval imports**, replaying emitted `VersionedCertificateArtifact` bundles from scratch (`25 / 25` Phase 8 + `80 / 80` `Gate 7A` + `12 / 12` `Gate 7B` + `80 / 80` `Gate 8A` = **`197 / 197` (`100.0%`) verified**).

---

## 1. Executive Summary, Dual-Host 64-Test Scorecard & Standalone Reconciliation

### 1.1 Unified 64-Test `pytest` Suite & Standalone Benchmark Ledgers Across Two Independent Hosts

| Verification Environment & Workload | Runtime Stack (`1` CPU Core, `cpu/wall ≈ 1.00`) | Result | `ruff` / Verifier | Wall / CPU Clock (`Peak RSS`) | Reconciliation Note |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Host A — Full `64`-Test `pytest` Suite** | `arm64`, Python `3.13.2`, PyTorch `2.10.0`, SciPy `1.16.3`, `highspy 1.15.1`, `coacd 1.0.7` | **`64 / 64` PASSED** | `0` warnings / `self_test=True` | **`950.2 s`** (`15m 50s`) | Live execution of all `14` test modules including [`test_p10_gate8_open_catalog_retrieval.py`](../tests/test_topo_brick/test_p10_gate8_open_catalog_retrieval.py). |
| **Host B — Full `64`-Test `pytest` Suite** | `arm64` (M4 Pro), Python `3.13.14`, PyTorch `2.15.0`, SciPy `1.18.1`, `highspy 1.15.1`, `coacd 1.0.7` | **`64 / 64` PASSED** | `0` warnings / `self_test=True` | **`748.0 s`** (`12m 28s`) | `100.0%` pass agreement across all `64` tests and all `10` Acceptance Gates (`Gates 1–5`, `6, 7A, 7B`, `8A, 8B`). |
| **Standalone `Phase 10 Gates 8A & 8B` CLI** | `python -m topo_brick.evaluation.phase10_open_catalog_retrieval_gates` (`OMP_NUM_THREADS=1`) | **`2 / 2` Phase 10 Gates PASSED** | `80 / 80` indep. certs verified | **B:** `Gate 8A: 101.9 s` (`424.2 MB`), `8B: 55.2 s` (`715.7 MB`)<br>**A:** `Gate 8A: 135.5 s` (`512.6 MB`), `8B: 73.3 s` (`901.9 MB`) | Archived in [`phase10-gate-verification-ledger.json`](phase10-gate-verification-ledger.json); **bit-for-bit (`0.000000`) cross-host quality, part-ID & `ΔE_00` match**. |
| **Standalone `Phase 9 Gates 6, 7A, 7B` CLI** | `python -m topo_brick.evaluation.phase9_compiler_and_hierarchical_gates` (`paired_ablation_count=2`) | **`3 / 3` Phase 9 Gates PASSED** | `92 / 92` indep. certs verified | **B:** `Gate 6: 7.48 s` (`395.3 MB`), `7A: 677.3 MB`, `7B: 715.4 MB`<br>**A:** `Gate 6: 9.82 s` (`515.1 MB`), `7A: 792.8 MB`, `7B: 905.0 MB` | Archived in [`phase9-gate-verification-ledger.json`](phase9-gate-verification-ledger.json); exact `0.0000` cross-host quality/coverage match. |
| **Standalone `S2B-600` Run (`2,400` builds/host)** | `S1`, `S2`, `S1_Reference_Assisted`, `Legolization_Baseline` (`highspy` `leaves=64, seed=0`) | **Exact Blind Match** (`597`, `598`, `18`) | Archived in `untouched_s2b_600_results.json` | **B:** `3,355.1 s` / `3,354.9 s`<br>**A:** `4,013.1 s` / `4,012.2 s` | `test_ws4` inside `pytest` audits this saved JSON + runs a live `12`-object stratified slice (`records[::50]`). |
| **Standalone `Phase 8 Gates 1–5` CLI** | `python -m topo_brick.evaluation.phase8_round9_gates` + live external `legolization==0.5.0` & `bricklayer-sdk==0.1.0` | **`5 / 5` Gates PASSED** | `25 / 25` indep. certs verified | **B:** `55.8 s` (`839.8 MB`)<br>**A:** `270.1 s` (`991.7 MB`) | Host A timing includes `160` live out-of-process `.stl -> .ldr` subprocess runs of `hbmartin/legolization`. |

---

### 1.2 Contract & Evidence Truth Matrix (`Post-Phase 10`, All `10 / 10` Gates Verified)

| Subsystem / Gate | Intended Contract | Implemented Mechanism | Observed Dual-Host Result (`Host B` / `Host A`) | Remaining Limitation / Boundary | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Open-Catalog `Top-K` Blind Sculpting (`Gate 8A` & `Gate 8B`)** | Bridge passive `16,085`-part catalog representation and active `realize()` synthesis to push original-mesh IoU past `0.92+` (`Bands 2–4` & `80`-mesh median) with `< 150 ms` B `p50` + commercial license & chromatic/studless controls. | [`catalog_retrieval_index.py`](../src/topo_brick/interface/catalog_retrieval_index.py) (`6,825` System molds, `223.7 KB`), [`open_catalog_realizer.py`](../src/topo_brick/kernel/open_catalog_realizer.py), [`open_catalog_blind_portfolio.py`](../src/topo_brick/kernel/open_catalog_blind_portfolio.py). | **`Gate 8A` (`n=80`):** `80/80` cert & openings clear & indep. verified; **`0.9064` overall mean / `0.9295` overall median `5×5×6` IoU (`62.23%` discretization loss reduction)**; **`Bands 2–4` mean `0.9349` (`100%` qual@0.90)**; `Band 1` mean `0.8209` (`0.9028` med); **`92.50%` qual@0.85**; **`46` distinct molds across `9 / 9` sculptural classes**; `1.39 ms` B / `2.00 ms` A query `p50`; **`133.79 ms` B / `205.11 ms` A end-to-end `p50`**.<br>**`Gate 8B`:** Monotonic `K`-ablation (`+0.0560` over `Gate 7A`, `52` molds at `Top-64`); `CC-BY-4.0` mode emits **`0` `CC-BY-SA-4.0` parts** (`40/40` cert, `0.8509` IoU); Option 3B **`ΔE_00 = 5.54` mean**, **`20/20` `.mpd` round-trip**, **`87.52%` exposed-stud reduction (`-0.0003` ΔIoU)**. | Maps caller-supplied `vertex_colors_rgb` and `semantic_style_hint` deterministically; does not claim cloud VLM open-ended text-to-3D aesthetic imagination. | **Verified** |
| **2. Automated `16,085`-Part LDraw + Shadow + `CoACD` Catalog (`Gate 6`)** | Exceed `BrickBench` (`14,441`) & `BrickNet` (`14,442`) with bore-preserving colliders, 6-DoF wrenches, and strict license/provenance separation. | [`ldraw_shadow_catalog_compiler.py`](../src/topo_brick/interface/ldraw_shadow_catalog_compiler.py) (`LDraw 2610` + `LDCad Shadow` + `coacd.run_coacd` + `PolytopeOBBSAT` + `COMPILED_LDRAW_CATALOG`). | **`16,072` compiled standalone / `16,085` combined parts** (`6,828` `CoACD` non-convex parts across `320` molds); **`500 / 500` (`100%`)** wall-overlap rejection; **`96.65%`** (`346/358`) exterior mate acceptance across `14` connector types; **`99.52%` OMR part coverage** (`26/26 >= 98%`, up from `93.85%`, `1.000` SE(3) round-trip across `18,322` connections). | `TIER_B` LDCad Shadow connectors carry `CC-BY-SA-4.0` (`SHARE_ALIKE_ADAPTATION`), filterable via `license_filter="cc_by_4_0_only"`. | **Verified** |
| **3. Fast `27`-Part Adaptive Brick + Plate + Slope Skinning (`Gate 7A`)** | Provide an ultra-fast `< 50 ms` B fixed-kit curved-mesh operating point cutting coarse discretization loss while preserving `100%` blind 6-DoF certification. | [`AdaptivePlateSlopeSkinning`](../src/topo_brick/kernel/adaptive_surface_and_macro_planner.py) (`surface_refinement="adaptive_plate_slope"` inside `realize()`). | Across all `80` frozen `Gate 2` unseen meshes: **`80 / 80` certified**, **`80 / 80` openings clear**, **`80 / 80` `independent_verifier` pass**, `5×5×6` original-mesh IoU **`0.7521 -> 0.8503` (`+0.0982` lift, `0.8825` median, `39.61%` discretization loss reduction)**, `qualified@0.80 = 83.75%` (up from `32.50%`), `22` part types, **`48.71 ms` B / `69.14 ms` A `p50`**. | Constrained by the `0.8668` `1-Stud-XY` Nyquist ceiling; pass `part_retrieval="open_catalog_top_k", surface_refinement="open_catalog_sculpted"` (`Gate 8A`) for `0.9064` mean / `0.9295` median IoU. | **Verified** |
| **4. Sub-Second `263–952`-Piece Macro-Block Planning (`Gate 7B`)** | Scale blind certified `realize()` from `<= 250` bricks to `250–1,000+` pieces under `< 1,000 ms` single-core median latency. | [`HierarchicalMacroBlockPlanner`](../src/topo_brick/kernel/adaptive_surface_and_macro_planner.py) (`macro_block_mode="auto"`, boundary seam keys, `561.7x` plumb hop cache + HiGHS fallback). | Across `36` unseen `263–952`-piece megastructures: **`36 / 36` certified** (`0.9994` mean target IoU), **`12 / 12` `independent_verifier` pass**, overall **`324.05 ms` B / `391.53 ms` A `p50`**, `M3_751_1100` **`415.34 ms` B / `548.05 ms` A `p50`** (`830.09 ms` B / `964.63 ms` A overall max!), **`520.41x` B / `1,786.54x` A paired speedup** vs. `macro_block_mode="off"`. | Plumb hop witness accelerates grounded vertical compression piers; non-plumb cantilevers/arches automatically fall back to exact sparse `highspy` LP. | **Verified** |
| **5. Blind Coarse Mesh & Voxel `realize()` (`Gate 2` & `S2B-600`)** | Synthesize opening-clear, certified builds from target geometry + kit + inventory + support/foundation policy alone. | [`TopoBrickKernel.realize`](../src/topo_brick/kernel/construction_kernel.py) (`S1` & `S2`, deterministic `highspy`, residual-verified witness reuse). | **`Gate 2` (`80` meshes, coarse):** `S2_NoSupport` **`80/80` cert, `83.75%` qual@0.70, `52.9 ms` B / `66.1 ms` A** (**`5.22x` / `4.95x` faster** than external `legolization --strategy luo` at `276.0` / `327.5 ms`, `33.75%` qual@0.60); **`S2B-600`:** **`598/600` (`S2`)** & **`597/600` (`S1`)** identical on both hosts. | Default `surface_refinement="none"` preserves legacy `20×20×24 LDU` coarse voxel semantics (`0.7531` mesh IoU). | **Verified** |
| **6. Falsification-Hardened `certify()` & Replay (`Gate 1 & 3`)** | Reject multi-joint saddles, internal-wall/depth collisions, intermediate `REORIENT`/`REMOVE_TEMP_SUPPORT` collapses; verify without planner code. | Coupled Hessian $\lambda_{\min}(H_{\text{free}})$, connector-specific socket envelopes, residual-checked primal witnesses, AST-isolated [`independent_verifier.py`](../src/topo_brick/evaluation/independent_verifier.py). | **`12 / 12` `Eval B-Extended` falsification cases `PASS`** ($\lambda_{\min}=-78.80$ on Case #7); **`200 / 200`** match vs. `2 LDU` CSG bore-hole collider (`0` FP, `0` FN); **`197 / 197`** total independent certificates replayed (`5.06e-12 g` max residual). | Insertion verifier checks monotone single-axis corridors rather than general 6-DoF spatial path planning. | **Verified** |
| **7. Real `BrickLayer` Agent A/B & LLM Masking (`Gate 5`)** | Lift external pretrained/code-emitting brick agents under matched prompts & budgets. | Live CMU `bricklayer-sdk==0.1.0` (`AvaLovelace1/brickwright`, MIT) paired harness (`n=50`) + `BrickGrammarLogitsProcessor` (`vocab=128,256`). | **`BrickLayer + TopoBrickKernel`:** **`47 / 50` (`94.0%`, CI `[83.5%, 98.8%]`)** native & 6-DoF qualified (`151.2` tokens, `20.5 ms` B tool p50) vs. **`0 / 50` (`0.0%`, `467.9` tokens)** standalone `BrickLayer`. | Evaluates local deterministic constructive code policy + compact `LlamaForCausalLM` rather than paid cloud LLM APIs. | **Verified** |
| **8. `24`-Cell Generative Versatility & Domain `repair()` (`Gate 3 & 4`)** | Blindly synthesize across `6` connector families × `4` conditions (`120` unseen tasks) & reserve `PROVEN_INFEASIBLE` for domain proofs. | Surface-normal pitch estimator + SNOT/clip-bar/pin-axle/Gauss-Newton loop synthesis + Farkas dual `_prove_repair_domain_infeasible`. | **`120 / 120` (`100.0%`, CI `[96.97%, 100.0%]`)** Blind Synthesis & Kernel Certified (`2.49e-14°` pitch err) vs. `19/120` `LuoInspired` & `20/120` `legolization`; Case #12 `1`-step $\to$ `INCONCLUSIVE`, `3`-step $\to$ `COMPLETED_CERTIFIED`. | `PhysicalValidationCert` is `VERIFIED` (`20 / 120`) only on the calibrated stud family; non-stud families remain `UNKNOWN` (`0 / 100`) pending wet-lab pull rigs. | **Verified (Scoped)** |
| **9. Audited Physics & Selective Learning Controls (`Gate 1 & 5`)** | Separate true hardware JSONs from solver fixtures, report motif-disjoint `2×2` confusion, and gate learned modules against strong deterministic controls. | `8` pure CMU hardware JSONs + `1` `16`-brick fixture + motif-disjoint `Train(60)->Holdout(120)` + `M3_Pairwise_Sparse` / `M1` controls. | **`8 / 8`** CMU hardware + **`1 / 1`** fixture; Holdout (`N=120`): `TP=72, TN=40`, **`FP=0` (`0.0%`)**, `FN=8` (`93.33%` acc); `M3_Pairwise_Sparse = 60/60` (`10.66 ms` B), kept opt-in behind deterministic `S1_HandcraftedGeomSynergy` (`5.34 ms` B) & `S1_SequentialConditional` (`1.68 ms` B). | Deterministic `S1` & `S2` remain the global production defaults (`0` learned parameters). | **Verified** |

---

## 2. Complete Verified Test Suite Ledger (`64` Tests Across `14` Modules)

*(Per-module timings verified under strict single-core pinning `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR=1` across Host A and Host B.)*

| # | Test Module | Tests | Host A (`py3.13.2` / `torch 2.10`) | Host B (`py3.13.14` / `torch 2.15`) | Primary Domain Covered |
| :-: | :--- | :-: | :--- | :--- | :--- |
| 1 | [`test_p10_gate8_open_catalog_retrieval.py`](../tests/test_topo_brick/test_p10_gate8_open_catalog_retrieval.py) | `6` | **`6 / 6` PASS** (`212.15 s`) | **`6 / 6` PASS** (`159.45 s`) | **Phase 10 (`Gates 8A & 8B`):** `6,825`-mold `uint64` retrieval index (`1.39 ms` B / `2.00 ms` A `p50`, nested `Top-16 ⊂ Top-32 ⊂ Top-64`), strict `CC-BY-4.0` filter (`0` ShareAlike parts), Option 3B `CIEDE2000` (`ΔE_00 = 5.54`) + `87.52%` studless tile reduction + `.mpd` round-trip, AST-isolated verifier index SHA-256 tamper rejection, `80`-mesh `Gate 8A` (`0.9064` mean / `0.9295` med / `0.9349` `Bands 2–4` IoU, `46` molds across `9` classes, `80/80` indep. certs), and `Gate 8B` `K`-ablation |
| 2 | [`test_p9_gates6_and_7_compiler_and_hierarchical.py`](../tests/test_topo_brick/test_p9_gates6_and_7_compiler_and_hierarchical.py) | `6` | **`6 / 6` PASS** (`351.31 s`) | **`6 / 6` PASS** (`273.25 s`) | **Phase 9 (`Gates 6, 7A, 7B`):** `16,072`-part compiled LDraw + Shadow + `CoACD` catalog & license separation, `500`-part kinematic audit + `99.52%` `26`-set OMR coverage, `80`-mesh adaptive plate+slope skinning (`0.8503` IoU, `80/80` indep. certs), `36` `263–952`-piece sub-second macro-block builds, hollow-rim vs. internal-wall collision, catalog SHA-256 tamper rejection |
| 3 | [`test_r9_gates1_to_5_phase8_verification.py`](../tests/test_topo_brick/test_r9_gates1_to_5_phase8_verification.py) | `5` | **`5 / 5` PASS** (`34.50 s`) | **`5 / 5` PASS** (`28.02 s`) | **Phase 8 (`Round 9 Gates 1–5`):** 1-core audit + `8/8` CMU + `16`-brick fixture + `2×2` confusion + planner-free verifier, `80`-mesh Pareto vs. `legolization`, `12` falsification cases + `200`-case CSG audit, `24`-cell matrix, real `BrickLayer` SDK + `M3`/`M1` controls |
| 4 | [`test_r8_ws1_to_ws6_phase7_verification.py`](../tests/test_topo_brick/test_r8_ws1_to_ws6_phase7_verification.py) | `5` | **`5 / 5` PASS** (`20.29 s`) | **`5 / 5` PASS** (`16.49 s`) | Phase 7 (`R8-WS1`–`R8-WS6`): `6/6` `Eval B`, `40` meshes + 6-term $T_{\text{total}}$, `7`-row `SE(3)` + loop closure, 3-tier physics, `M3_Pairwise` + `Eval C` |
| 5 | [`test_revised_construction_kernel.py`](../tests/test_topo_brick/test_revised_construction_kernel.py) | `4` | **`4 / 4` PASS** (`35.75 s`) | **`4 / 4` PASS** (`28.81 s`) | Round 7 Deliverables 1–4: kernel invariants, `21` untouched external objects, cached adapter/repair, dynamic gates |
| 6 | [`test_wp_l10_l11_l12_stage5.py`](../tests/test_topo_brick/test_wp_l10_l11_l12_stage5.py) | `1` | **`1 / 1` PASS** (`40.34 s`) | **`1 / 1` PASS** (`32.40 s`) | Continuation-preserving cache, hybrid router + continuous McCormick MILP, Stage 5 shapes |
| 7 | [`test_wp_l13_l14_l15_l16_stage55_and_stage6.py`](../tests/test_topo_brick/test_wp_l13_l14_l15_l16_stage55_and_stage6.py) | `1` | **`1 / 1` PASS** (`62.02 s`) | **`1 / 1` PASS** (`49.23 s`) | Auditable run bundle, 4-way target partition invariance (`0.0` diff), 16-object pilot, 6-planner study |
| 8 | [`test_wp_l1_interface.py`](../tests/test_topo_brick/test_wp_l1_interface.py) | `8` | **`8 / 8` PASS** (`< 0.04 s`) | **`8 / 8` PASS** (`< 0.03 s`) | `SO(3)_cubic` group, persistent port manifold, whole-hull & SNOT `5:5:6` cycles |
| 9 | [`test_wp_l2_fields.py`](../tests/test_topo_brick/test_wp_l2_fields.py) | `3` | **`3 / 3` PASS** (`1.03 s`) | **`3 / 3` PASS** (`0.78 s`) | 24-ch frontier & 18-ch candidate tensors, quadrant feasibility, $SE(3)$ invariance |
| 10 | [`test_wp_l3_planner_and_equilibrium.py`](../tests/test_topo_brick/test_wp_l3_planner_and_equilibrium.py) | `4` | **`4 / 4` PASS** (`3.34 s`) | **`4 / 4` PASS** (`2.67 s`) | Buried-joint LP overload backoff, cantilever torque, `IPA`/`GNN` latency, Stage 1 wall |
| 11 | [`test_wp_l4_l5_diffusion_and_benchmarks.py`](../tests/test_topo_brick/test_wp_l4_l5_diffusion_and_benchmarks.py) | `4` | **`4 / 4` PASS** (`2.59 s`) | **`4 / 4` PASS** (`2.03 s`) | Interaction graph (`E_conflict`, `E_synergy`), Benchmarks `A1`, `A2`, `B`, `C`, denoiser grads |
| 12 | [`test_wp_l55_l6_l7_stage25_and_stage3.py`](../tests/test_topo_brick/test_wp_l55_l6_l7_stage25_and_stage3.py) | `6` | **`6 / 6` PASS** (`34.47 s`) | **`6 / 6` PASS** (`27.53 s`) | McCormick MILP, multi-face SNOT, $C_4$ equivariance, LDraw round-trip, `WP-L6`, incremental cache |
| 13 | [`test_wp_l75_l8_l9_stage35_and_stage4.py`](../tests/test_topo_brick/test_wp_l75_l8_l9_stage35_and_stage4.py) | `6` | **`6 / 6` PASS** (`133.51 s`) | **`6 / 6` PASS** (`101.94 s`) | 2-course teacher, strategy mixing, D3PM paired ablation, stability ridge, `IntegralTargetGrid`, Stage 4 |
| 14 | [`test_ws1_to_ws5_complete_verification.py`](../tests/test_topo_brick/test_ws1_to_ws5_complete_verification.py) | `5` | **`5 / 5` PASS** (`112.96 s`) | **`5 / 5` PASS** (`88.91 s`) | `WS-1`–`WS-5` (`Claims A–F`): sparse tiles, blind `realize`, CMU `8/8`, `TIER_A` + `26` OMR + Set `10698` (`5/5`), reconciled `S2B-600` + `12`-obj live slice + `30` Overhangs, `M2` + live `Llama` |
| **Total** | **All 14 Modules** | **`64`** | **`64 / 64` PASS (`950.2 s`)** | **`64 / 64` PASS (`748.0 s`)** | **Complete `topo_brick` Phase 1–10 dual-host verification & All `10 / 10` Acceptance Gates** |

---

## 3. Verified Acceptance Gate Results (`Gates 1–5` + Phase 9 `Gates 6, 7A, 7B` + Phase 10 `Gates 8A, 8B`)

### 3.1 `Gate 1` — Evidence Audit, True 1-Core Pinning, Hardware Provenance & Reconciled `S2B-600`
1. **True Single-Core Audit (`enforce_and_audit_single_core`):** Verified `OMP=1`, `OPENBLAS=1`, `MKL=1`, `VECLIB=1`, `NUMEXPR=1`, `torch.get_num_threads()=1`, `highs_threads=1` with `cpu_to_wall_ratio = 0.9999` (Host B) and `1.0000` (Host A).
2. **CMU `StableLego` Provenance Split & Motif-Disjoint `2×2` Confusion Matrix:**
   * **Hardware-Labeled CMU Boundary JSONs (`n = 8`):** **`8 / 8` (`100.0%`)** matched (`external_weight_fail/good`, `stair_19/20/20_good`, `stick_heavy/good/light`).
   * **Upstream `16`-Brick Horizontal-Contact Solver Fixture (`n = 1`):** `stick_heavy_good_test_horizontal_force.json` (`sha1 e5c1175606c447e10600f80a89f04b3d99374a82`, `16` bricks under `1g` gravity testing side-wall normal contact): **`VERIFIED` (`STABLE`)**.
   * **Synthetic Lateral-Push Sanity Check (`n = 1`):** `0.45 g` horizontal load failure check (`VERIFIED`, strictly separated from CMU hardware).
   * **Motif-Disjoint Holdout `2×2` Confusion Matrix (`Train N = 60` on 3 families $\to$ `Holdout N = 120` on 3 unseen families: `asymmetric_corbel_wall`, `twin_pier_cantilever_bridge`, `overhung_corner_L_balcony`):**
     * `TP = 72`, `TN = 40`, **`FP = 0`**, `FN = 8` (`Accuracy = 93.33%`);
     * **`FP / N_total`:** `0 / 120 = 0.00%` (95% CI `[0.00%, 3.03%]`);
     * **`FP / (FP + TN)` (`1 - Specificity` on unstable ground truth):** `0 / 40 = 0.00%` (95% CI `[0.00%, 8.81%]`);
     * **`FP / (TP + FP)` (`False Discovery Rate` on accepted builds):** `0 / 72 = 0.00%` (95% CI `[0.00%, 4.99%]`);
     * **`FN / (TP + FN)` (Conservative rejection rate):** `8 / 80 = 10.00%` (95% CI `[4.42%, 18.76%]`).
3. **Standalone `S2B-600` (`n = 600` Unique `object_id`s, `2,400` Realizations/Host):**

| Solver & Mode (`n = 600`: `450` $\le 100$ + `150` $101..250$) | Certified (`Host B` & `Host A`) | 95% Clopper-Pearson CI | Mean Voxel IoU | Host B / Host A `p50` (`<=100` bricks) | Host B / Host A `p50` (`All 600`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`S2_Blind_WholeTarget` (`PairSeededLocalSearch`)** | **`598 / 600` (`99.67%`, Identical)** | **`[98.80%, 99.96%]`** | **`0.9956`** | **`591.09 ms` / `706.36 ms`** | **`879.74 ms` / `1,013.69 ms`** |
| **`S1_Blind_WholeTarget` (`CoverAndRepairSolver`)** | **`597 / 600` (`99.50%`, Identical)** | **`[98.55%, 99.90%]`** | **`0.9952–0.9963`** | **`610.92 ms` / `730.44 ms`** | **`856.01 ms` / `1,011.94 ms`** |
| **`S1_Reference_Assisted` (Dataset Audit)** | `505 / 600` B · `497 / 600` A | `[81.0%, 87.0%]` B | `0.9545–0.9567` | — | — |
| **`Legolization_Baseline_Blind` (`GreedyLayerMerge`)** | **`18 / 600` (`3.00%`, Identical)** | `[1.79%, 4.70%]` | `1.0000` | `32.92 ms` / `44.00 ms` | `36.48 ms` / `49.27 ms` |

---

### 3.2 `Gate 2` — Frozen Unseen Continuous Triangle-Mesh Benchmark (`n = 80` Across 4 Size Bands) vs. Pinned External `hbmartin/legolization` (`legolization==0.5.0`)

*(Selection funnel: `82` candidates $\to$ `2` non-manifold excluded $\to$ `80` watertight unseen meshes, `20` per band across `<=25`, `26–60`, `61–120`, `121–250` bricks; `40` meshes carry required architectural openings; fine `5×5×6 LDU` original-triangle-mesh ground truth incurs `mean_voxelization_iou_loss = 0.2477` on `20×20×24 LDU` coarse bricks — closed in `Gate 7A` §3.7 below.)*

| Solver / External Comparator (`n = 80` Unseen Meshes) | Certified (`95% CI`) | Openings Clear | Qualified `@0.60` / `@0.70` | Mean Orig-Mesh IoU (Chamfer LDU) | Host B / Host A `p50` (`p95`) ms | Band `1 / 2 / 3 / 4` `p50` ms (`Host B`) | Scaling (`Ports / Cands / LP / Wit`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`S2_NoSupport` (`PairSeededLocalSearch`)** | **`80/80` (`100%`)**<br>`[95.5%, 100%]` | **`80 / 80`** | **`83.75%` / `83.75%`**<br>*(`93.75%` @0.50)* | **`0.7531`** (`9.87`) | **`52.86` / `66.11`**<br>(`332.9` / `427.6`) | **`12.6 / 23.2 / 94.7 / 323.5`** | `40.6 / 1160 / 1.73 / 44.9` |
| **`S1_PermittedSupport` (`CoverAndRepair`)** | **`80/80` (`100%`)**<br>`[95.5%, 100%]` | **`80 / 80`** | **`83.75%` / `83.75%`** | **`0.7521`** (`9.85`) | **`68.83` / `112.35`**<br>(`442.4` / `723.2`) | **`15.4 / 26.8 / 118.2 / 352.7`** | `40.6 / 1161 / 0.11 / 41.5` |
| **`S1_NoSupport` (`CoverAndRepair`)** | **`80/80` (`100%`)**<br>`[95.5%, 100%]` | **`80 / 80`** | **`83.75%` / `83.75%`** | **`0.7529`** (`9.84`) | **`73.20` / `117.53`**<br>(`465.2` / `663.2`) | **`19.5 / 30.8 / 120.7 / 401.1`** | `40.6 / 1161 / 2.11 / 40.9` |
| **`LuoInspiredSeamRefinement` (Internal)** | `69/80` (`86.3%`)<br>`[76.7%, 92.9%]` | `80 / 80` | `71.25%` / `71.25%` | `0.7523` (`9.87`) | `7.96` / `10.05`<br>(`28.4` / `36.2`) | `2.4 / 4.0 / 22.1 / 27.1` | `0 / 0 / 0.26 / 25.5` |
| **`GreedyLayerMerge_Control` (Internal)** | `16/80` (`20.0%`)<br>`[11.9%, 30.4%]` | `80 / 80` | `8.75%` / `8.75%` | `0.7523` (`9.87`) | `8.06` / `9.99`<br>(`28.1` / `35.5`) | `2.7 / 3.8 / 11.4 / 24.8` | `0 / 0 / 0.44 / 47.0` |
| **`External_Legolization_Luo` (`0.5.0`, CLI)** | `49/80` (`61.3%`)<br>`[49.7%, 71.9%]` | `40 / 80` | `33.75%` / `12.50%` | `0.5998` (`11.68`) | `276.0` / `327.5`<br>(`988.0` / `1176.1`) | `51.5 / 111.0 / 305.5 / 986.5` | Out-of-process `.stl -> .ldr` |
| **`External_Legolization_Greedy` (`0.5.0`, CLI)** | `69/80` (`86.3%`)<br>`[76.7%, 92.9%]` | `40 / 80` | `33.75%` / `12.50%` | `0.5998` (`11.68`) | `196.0` / `236.0`<br>(`901.0` / `1075.0`) | `79.0 / 138.0 / 275.5 / 769.0` | Out-of-process `.stl -> .ldr` |

---

### 3.3 `Gate 3` — `Eval B-Extended` 12-Case Certificate Falsification Suite & `2 LDU` CSG Audit
* **`12 / 12` Adversarial Falsification Cases Passed (`Host A` & `Host B`):** Original Cases 1–6 plus Case #7 coupled 2-joint pantograph saddle rejected ($\lambda_{\min}(H_{\text{free}}) = -78.80 < 0$), Case #8 same-part internal-wall socket collision rejected, Case #9 invalid axial insertion depth rejected, Case #10 `REORIENT_ASSEMBLY` intermediate $90^\circ$ gravity collapse rejected, Case #11 premature `REMOVE_TEMP_SUPPORT` collapse rejected, and Case #12 bounded 1-step `repair()` returning **`INCONCLUSIVE`** while 3-step search certifies **`COMPLETED_CERTIFIED`**.
* **Independent `2 LDU` (`0.8 mm`) CSG Bore-Hole Collider Audit (`n = 200`):** **`0 / 200` false acceptances and `0 / 200` false rejections**.

---

### 3.4 `Gate 4` — Frozen `6 × 4 = 24`-Cell End-to-End Generative Versatility Matrix (`n = 120` Unseen Tasks)

| Connector / Construction Family (`4` Conditions × `5` Tasks = `20` / Row) | Import Verified | Blind Synthesis | Kernel Certified (`95% CI`) | Physically Validated | `LuoInspired` Cert | External `legolization` Cert |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. `snot_and_lateral_offsets`** | **`20 / 20`** | **`20 / 20`** | **`20 / 20` (`[83.2%, 100%]`)** | `0 / 20` (`UNKNOWN`) | `9 / 20` | `10 / 20` |
| **2. `clips_and_bars`** | **`20 / 20`** | **`20 / 20`** | **`20 / 20` (`[83.2%, 100%]`)** | `0 / 20` (`UNKNOWN`) | `0 / 20` | `0 / 20` |
| **3. `hinged_roofs_and_gates`** *(held-out `13°,19°,27°,34°,38°`; max err `2.49e-14°`)* | **`20 / 20`** | **`20 / 20`** | **`20 / 20` (`[83.2%, 100%]`)** | `0 / 20` (`UNKNOWN`) | `0 / 20` | `0 / 20` |
| **4. `axles_and_pins`** *(smooth vs. friction pin vs. cross-axle)* | **`20 / 20`** | **`20 / 20`** | **`20 / 20` (`[83.2%, 100%]`)** | `0 / 20` (`UNKNOWN`) | `0 / 20` | `0 / 20` |
| **5. `closed_kinematic_loops`** *(Gauss-Newton loop closure + monotone trace)* | **`20 / 20`** | **`20 / 20`** | **`20 / 20` (`[83.2%, 100%]`)** | `0 / 20` (`UNKNOWN`) | `0 / 20` | `0 / 20` |
| **6. `mixed_subassemblies_and_retail_inventory`** *(Set `10698-1` + `PLACE_SUBASSEMBLY`)* | **`20 / 20`** | **`20 / 20`** | **`20 / 20` (`[83.2%, 100%]`)** | **`20 / 20` (`PASS`)** | `10 / 20` | `10 / 20` |
| **Total Across All `24` Cells (`n = 120` Unseen Tasks)** | **`120 / 120`** | **`120 / 120`** | **`120 / 120` (`[96.97%, 100%]`)** | **`20 / 120` (`16.67%`)** | `19 / 120` (`15.8%`) | `20 / 120` (`16.7%`) |

---

### 3.5 `Gate 5` — Real CMU `BrickLayer` SDK Paired Agent Study (`n = 50`) & Unseen-Family `M3` / `M1` Controls
* **Paired Constructive Code Agent on Real `bricklayer-sdk==0.1.0` (`AvaLovelace1/brickwright`, MIT, `n = 50` Tasks):**
  * **`BrickLayer_Standalone` (`is_valid()` + `is_connected()`):** **`0 / 50` (`0.0%`, CI `[0.0%, 7.1%]`)** native & qualified builds (`467.88` mean tokens, `2.52` model calls, `1.52` retries).
  * **`BrickLayer + TopoBrickKernel` (`constrain` + witness-verified prefix `certify` + `repair`):** **`47 / 50` (`94.0%`, CI `[83.45%, 98.75%]`)** native `BrickLayer` valid **and `47 / 50` (`94.0%`) full 6-DoF prefix + opening-clear qualified builds** (`151.20` mean tokens, `2.00` model calls, `20.53 ms` B / `25.95 ms` A median tool latency).
* **Unseen-Family `M3_Pairwise_Sparse` (`n = 60` across `3` unseen families) & `M1` Controls:**
  * `M3_Pairwise_Sparse` (`0.495` sparse edge density, `0.51 s` train wall, `19`-build break-even vs. `S1_ExactTwoCourse`) achieves **`60 / 60` (`1.000`)** in **`10.66 ms` B / `13.78 ms` A** (**`3.02x` / `2.99x` faster** than `S1_ExactTwoCourse` at `32.15 ms` B / `41.26 ms` A) vs. `0.000` for `S1_Myopic`, `S2`, and `S1_CoherentTieBreak`—while deterministic `S1_HandcraftedGeomSynergy` (`60 / 60` in **`5.34 ms` B / `6.76 ms` A**) and `S1_SequentialConditional` (`60 / 60` in **`1.68 ms` B / `1.78 ms` A**) match `1.000` faster with `0` training overhead $\implies$ **`M3_Pairwise_Sparse` honestly kept opt-in (`promoted_to_default = False`)**.
  * `M1_Learned16D` (`24 / 24 = 1.000`) ties `RawMetadataControl` (`24 / 24 = 1.000`) across `4` held-out part families, confirming explicit declarative metadata suffices without learned projection weights.

---

### 3.6 `Gate 6` (`Phase 9`) — Automated `16,085`-Part LDraw `2610` + LDCad Shadow + `CoACD` Catalog Compiler & `26`-Set OMR Coverage
1. **Standalone Catalog Scale, Non-Convex `CoACD` Polytopes & Explicit License/Provenance Separation:**
   * **Catalog Size:** **`16,072` compiled standalone parts** (`COMPILED_LDRAW_CATALOG`) and **`16,085` combined parts** (`COMBINED_BRICK_CATALOG` with `213` curated `TIER_A_CATALOG` overrides), surpassing both `BrickBench` (`14,441`) and `BrickNet` (`14,442`) (`excluded_counts`: `3,586` stickers/non-system-scale, `2,563` obsolete/moved aliases, `1,987` internal `s/` helpers, `777` empty/no-port records).
   * **Provenance Tiers & Licensing:** `TIER_B_LDCAD_SHADOW_DAG = 15,105` and `TIER_B_BASE_MOLD_SHADOW = 49` (`geometry_license = "CC-BY-4.0"`, `connector_license = "CC-BY-SA-4.0"`, `redistribution_status = "SHARE_ALIKE_ADAPTATION"` from `RolandMelkert/LDCadShadowLibrary@9b1131fb`) strictly separated from `TIER_C_LDRAW_PRIMITIVE_GEOMETRIC = 918` and `TIER_A = 213` (`COMMERCIALLY_USABLE_WITH_ATTRIBUTION`, `CC-BY-4.0`).
   * **Real `coacd.run_coacd` Non-Convex Decomposition & `PolytopeOBBSAT`:** **`6,828` non-convex compiled parts** decomposed via `coacd.run_coacd` (`threshold = 0.055`, `max_convex_hull = 16`) across `320` base mold shells (`coacd_concave_pocket_clearance_ok = True`, `coacd_solid_overlap_rejected_ok = True`), paired with bore-carved `X -> Y -> Z` disjoint cuboid covers preserving cylindrical/cruciform socket clearances.
2. **Stratified `500`-Part Kinematic Audit & `26`-Set Official Model Repository (`.mpd`) Benchmark:**
   * **Kinematic & Wall-Collision Audit (`n = 500`):** **`500 / 500` (`100.0%`)** solid-wall overlaps rejected (`0` false acceptances) and **`346 / 358` (`96.65%`)** exterior socket mates accepted across **`14` exercised connector types** (`STANDARD_STUD`, `HOLLOW_STUD`, `BAR_3_18MM`, `CLIP_3_18MM`, `TECHNIC_PIN`, `TECHNIC_PIN_HOLE`, `SMOOTH_PIN`, `TECHNIC_AXLE`, `TECHNIC_AXLE_HOLE`, `HINGE_PIN`, `HINGE_SOCKET`, `LOCKING_HINGE_PIN`, `BALL_JOINT`, `BALL_SOCKET`) in `7.48 s` B (`395.3 MB` Peak RSS) / `9.82 s` A (`515.1 MB` Peak RSS).
   * **Human `26`-Set OMR `.mpd` Coverage:** Aggregate part-placement coverage lifted from `93.85%` (`TIER_A` baseline) to **`99.52%`** (`99.68%` mean per-set coverage, **`26 / 26` sets $\ge 98.0\%$**), with **`1.000` (`100.0%`) exact `SE(3)` round-trip fidelity** across **`18,322` compatible port connections** spanning `5` connector families (`STUD_TUBE_AND_SNOT`, `CLIP_AND_BAR`, `HINGE_TURNTABLE_BALL`, `TECHNIC_PIN_AND_WHEEL`, `TECHNIC_AXLE`).

---

### 3.7 `Gate 7A` (`Phase 9`) — Adaptive Brick + `8 LDU` Plate + Roof-Slope Surface Skinning (`n = 80` Unseen Meshes)

*(Evaluated on the exact frozen `80`-mesh unseen corpus and untouched `5×5×6 LDU` sub-voxel ground-truth evaluator from `Gate 2`, plus an auxiliary `5×5×2 LDU` sub-plate grid, with planner-free `verify_certificate_independently` executed on all `80 / 80` builds.)*

| Band (`20` Unseen Meshes / Band) | Coarse `20×20×24` Mean `5×5×6` IoU | Adaptive `Brick+Plate+Slope` Mean `5×5×6` IoU (`Median`) | Auxiliary `5×5×2` Sub-Plate Mean IoU | Certified & Openings Clear | `independent_verifier` Pass | Host B / Host A `p50` ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Band1_le25` (`<= 25` bricks)** | `0.5740` | **`0.7059`** (`0.7466`) | `0.6816` | **`20 / 20` & `20 / 20`** | **`20 / 20`** | **`21.43 ms` / `30.00 ms`** |
| **`Band2_26to60` (`26–60` bricks)** | `0.7624` | **`0.8672`** (`0.8821`) | `0.8677` | **`20 / 20` & `20 / 20`** | **`20 / 20`** | **`41.71 ms` / `59.04 ms`** |
| **`Band3_61to120` (`61–120` bricks)** | `0.8055` | **`0.9057`** (`0.9057`) | `0.8929` | **`20 / 20` & `20 / 20`** | **`20 / 20`** | **`70.48 ms` / `97.97 ms`** |
| **`Band4_121to250` (`121–250` bricks)** | `0.8663` | **`0.9223`** (`0.9223`) | `0.9225` | **`20 / 20` & `20 / 20`** | **`20 / 20`** | **`84.65 ms` / `117.25 ms`** |
| **All `80` Meshes (`Gate 7A` Total)** | `0.7521` (`0.7869` med)<br>*`32.50%` qual@0.80* | **`0.8503` (`0.8825` med, `+0.0982` lift)**<br>**`83.75%` qual@0.80 (`92.50%` @0.70)** | **`0.8412`** (`0.8822` med) | **`80 / 80` & `80 / 80`** | **`80 / 80` (`100%`)** | **`48.71 ms` / `69.14 ms`**<br>*(`p90`: `96.4` / `134.7 ms`; `max`: `113.4` / `157.6 ms`)* |

* **Discretization Loss Reduction & Part Diversity:** Cuts `Gate 2`'s `0.2477` coarse-brick discretization gap by **`39.61%`** (`discretization_loss_reduction_ratio = 0.3961`, `mean_chamfer_l2_ldu: 9.85 -> 9.68 LDU`) while actively placing **`22` distinct brick + plate + slope part types** (`2456`, `3001`–`3005`, `3008`–`3010`, `3020`–`3024`, `3039`, `3040b`, `3460`, `3622`, `3623`, `3710`, `54200`, `85984`) at `Peak RSS = 677.3 MB` B / `792.8 MB` A.

---

### 3.8 `Gate 7B` (`Phase 9`) — Hierarchical Macro-Block Planning Across `36` Unseen `263–952`-Piece Megastructures (`OMP=1`)

| Megastructure Band (`12` Unseen Targets / Band) | Realized Piece Range (`Mean`) | Certified (`95% CI`) | Mean Target IoU | Host B `p50` (`p90` / `max`) ms | Host A `p50` (`p90` / `max`) ms |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`M1_250_450` (`250–450` pieces)** | `263 – 384` (`321.6`) | **`12 / 12` (`100%`)** | **`0.9991`** | **`140.19 ms`** (`174.03` / `486.47 ms`) | **`184.25 ms`** (`221.30` / `581.25 ms`) |
| **`M2_451_750` (`451–750` pieces)** | `510 – 638` (`568.0`) | **`12 / 12` (`100%`)** | **`0.9990`** | **`282.31 ms`** (`632.26` / `672.46 ms`) | **`363.58 ms`** (`746.72` / `760.71 ms`) |
| **`M3_751_1100` (`751–1,100` pieces)** | `752 – 952` (`863.8`) | **`12 / 12` (`100%`)** | **`1.0000`** | **`415.34 ms`** (`797.76` / `830.09 ms`) | **`548.05 ms`** (`939.14` / `964.63 ms`) |
| **All `36` Megastructures (`Gate 7B` Total)** | **`263 – 952`** | **`36 / 36` (`[90.26%, 100%]`)** | **`0.9994`** | **`324.05 ms`** (`705.99` / **`830.09 ms`**) | **`391.53 ms`** (`825.44` / **`964.63 ms`**) |

* **Paired Macro-Block Speedup & Planner-Free Replay:** Paired `macro_block_mode="auto"` vs. `"off"` ablation (`n = 2` on `m1_00` & `m1_01`, archived in `phase9-gate-verification-ledger.json`) yields **`520.41x` speedup on Host B (`126.99 ms` vs. `66,087.26 ms`)** and **`1,786.54x` speedup on Host A (`170.19 ms` vs. `304,055.10 ms`)**, with **`12 / 12` (`100.0%`)** stratified megastructure certificates (`4` per band) independently verified by [`independent_verifier.py`](../src/topo_brick/evaluation/independent_verifier.py) (`Peak RSS = 715.4 MB` B / `905.0 MB` A).

---

### 3.9 `Gate 8A` (`Phase 10`) — Open-Catalog `Top-48` Blind Sculpting Benchmark (`n = 80` Unseen Meshes, Untouched `5×5×6 LDU` Evaluator)

*(Evaluated on `Gate 2`'s exact frozen `80`-mesh corpus and untouched `5×5×6 LDU` `evaluate_assembly_against_original_mesh` judge with `retrieval_top_k=48`, strict `OMP_NUM_THREADS=1` [`cpu/wall = 0.9999` B / `0.9982` A, `Peak RSS = 424.2 MB` B / `512.6 MB` A], and `80 / 80` planner-free `verify_certificate_independently` replays.)*

| Band (`20` Unseen Meshes / Band) | Coarse `5×5×6` IoU | `Gate 7A` (`27`-Part) IoU | **Open-Catalog Mean (`Median`) `5×5×6` IoU** | Auxiliary `5×5×2` IoU | `qualified@0.85` / `@0.90` | Certified & Indep. Replay | Host B / Host A End-to-End `p50` ms |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Band1_le25` (`<= 25` bricks)** | `0.5740` | `0.7059` | **`0.8209`** (`0.9028`) | `0.7546` | `70.0%` / `70.0%` | **`20 / 20` & `20 / 20`** | **`16.35 ms` / `28.33 ms`** |
| **`Band2_26to60` (`26–60` bricks)** | `0.7624` | `0.8672` | **`0.9181`** (`0.9187`) | `0.8860` | **`100.0%` / `100.0%`** | **`20 / 20` & `20 / 20`** | **`96.52 ms` / `128.99 ms`** |
| **`Band3_61to120` (`61–120` bricks)** | `0.8055` | `0.9057` | **`0.9349`** (`0.9349`) | `0.9218` | **`100.0%` / `100.0%`** | **`20 / 20` & `20 / 20`** | **`188.48 ms` / `413.62 ms`** |
| **`Band4_121to250` (`121–250` bricks)** | `0.8663` | `0.9223` | **`0.9516`** (`0.9516`) | `0.9453` | **`100.0%` / `100.0%`** | **`20 / 20` & `20 / 20`** | **`671.44 ms` / `727.24 ms`** |
| **`Bands 2–4` (`n = 60` Sculptural Meshes)** | `0.8114` | `0.8984` | **`0.9349` (`0.9297` med)** | `0.9177` | **`100.0%` / `100.0%`** | **`60 / 60` & `60 / 60`** | — |
| **All `80` Meshes (`Gate 8A` Total)** | `0.7521` | `0.8503` | **`0.9064` (`0.9295` med, `62.23%` loss cut)** | **`0.8769`** | **`92.50%` (`74/80`) `@0.85`** | **`80 / 80` & `80 / 80`** | **`133.79 ms` B / `205.11 ms` A**<br>*(Query `p50`: `1.39` / `2.00 ms`)* |

* **Open-Catalog Diversity (`46` Distinct Molds Across `9 / 9` Sculptural Classes):** Automatically retrieved from `6,825` indexed System molds (`compiled_mold_retrieval_index.npz`) and placed `46` distinct part IDs (`13548`, `15573`, `24307`, `2431`, `2456`, `25269`, `3001`–`3005`, `3010`, `3020`–`3024`, `3040a`, `30414`, `3043`, `3068b`, `3069b`, `3070b`, `3623`, `3659`, `3660`, `3665`, `3666`, `3678b`, `36841`, `3684c`, `3710`, `3795`, `4070`, `44675`, `44728`, `49307`, `49656`, `54200`, `63864`, `78886`, `79757`, `85984`, `87079`, `87087`, `99780`) exercising `ARCH`, `CURVED_SLOPE`, `INVERTED_SLOPE`, `JUMPER_HALF_STUD`, `PLANAR_SLOPE`, `SMOOTH_TILE`, `SNOT_ANCHOR`, `STEEP_SLOPE`, and `XY_WEDGE_ROUND` with **bit-for-bit identical cross-host outputs**.

---

### 3.10 `Gate 8B` (`Phase 10`) — Stratified `K`-Ablation (`n = 40`), Commercial `CC-BY-4.0` Mode & Option 3B Chromatic/Studless Study (`n = 20`)

1. **Monotonic Retrieval `K`-Ablation (`n = 40` Stratified Meshes, `40 / 40` Certified Across All 5 Arms):**
   * `Fixed_10_Coarse`: `0.7300` mean (`0.7869` med, `10` parts, `63.34 ms` B / `82.07 ms` A `p50`)
   * `Fixed_27_Gate7A`: `0.8243` mean (`0.8825` med, `22` parts, `36.05 ms` B / `51.20 ms` A `p50`)
   * `OpenCatalog_Top16`: `0.8580` mean (`0.9166` med, `17` parts, `71.01 ms` B / `94.63 ms` A `p50`)
   * `OpenCatalog_Top32`: `0.8640` mean (`0.9241` med, `28` parts, `131.78 ms` B / `179.53 ms` A `p50`)
   * **`OpenCatalog_Top64`:** **`0.8802` mean (`0.9236` med, `52` distinct molds, `167.25 ms` B / `226.35 ms` A `p50`)** — verifying strictly nested subsets (`Top-16 ⊂ Top-32 ⊂ Top-64`), monotonic retrieval lift, **`+0.0560` mean IoU lift over `Gate 7A`**, and **`+0.1502` lift over `Fixed-10`**.
2. **Strict Commercial `CC-BY-4.0` Mode (`license_filter="cc_by_4_0_only"`, `n = 40`):** **`40 / 40` (`100.0%`) certified**, **`0` `CC-BY-SA-4.0` parts emitted**, achieving **`0.8509` mean `5×5×6` IoU** strictly with `TIER_A` + `TIER_C` commercial-attribution molds.
3. **Option 3B Perceptual Chromatic & Studless-Tile Skinning (`n = 20` Multi-Region Colored Meshes):** Assigns LDraw solid-palette colors via `CIEDE2000` (`LDRAW_SOLID_PALETTE_SRGB`, **`ΔE_00 = 5.54` mean `<= 8.0`**) with **`20 / 20` exact `.mpd` Type-1 round-trip**, while `semantic_style_hint="smooth_studless"` cuts exposed upward studs from `569` $\to$ `71` (**`87.52%` reduction `>= 60%`**) at `20 / 20` 6-DoF certification and **`-0.0003` mean IoU delta**.

---

## 4. Honest Remaining Scope Boundaries (What Is *Not* Claimed)

1. **Physical Hardware Scope (`PhysicalValidationCert = UNKNOWN` Default on Non-Stud Connectors):** Hardware validation covers the `8` published hardware-labeled CMU `StableLego` JSONs (`8 / 8`), `1` upstream `16`-brick horizontal-contact fixture (`e5c11756`), and `180` calibrated stud boundary assemblies (`Train 60 -> Holdout 120`, `0` false passes). Non-stud families (`SNOT`, `clips/bars`, `hinges`, `axles/pins`, `closed loops`) are reported honestly as `0 / 100` (`UNKNOWN`) in `Gate 4`'s `Physically Validated` column pending wet-lab load-cell pull/torsion rigs.
2. **ShareAlike Licensing on `TIER_B` LDCad Shadow Metadata vs. `CC-BY-4.0` Commercial Mode:** The `16,085`-part combined catalog achieves `99.52%` part-placement coverage across the `26`-set OMR benchmark (the remaining `0.48%` consists of flexible hoses/strings, cloth sails, and custom sticker sheets). Every `TIER_B` part compiled from LDCad Shadow carries `connector_license = "CC-BY-SA-4.0"` (`SHARE_ALIKE_ADAPTATION`), while `license_filter="cc_by_4_0_only"` (`Gate 8B`) enforces a verified `0`-ShareAlike commercial mode (`TIER_A` + `TIER_C` only).
3. **Deterministic Chromatic/Style Mapping vs. Frontier Cloud VLM Open-Ended Prompting, & Monotone Insertion Corridors:** Option 3B (`Gate 8B`) deterministically maps caller-supplied 3D vertex colors (`CIEDE2000` `ΔE_00 = 5.54`) and `"smooth_studless"` tile finishes (`87.52%` stud reduction) onto 6-DoF-certified builds; we do **not** claim open-ended text-to-3D visual/semantic imagination over frontier cloud VLMs (whose generated meshes or code programs instead plug directly into `TopoBrickKernel` as demonstrated in `Gate 5`'s `94.0%` `BrickLayer` lift and `Gate 8A`'s `0.9295` median mesh sculpting). Finally, `certify()` and `independent_verifier.py` verify 6-axis monotone single-step insertion corridors (`±X, ±Y, ±Z` and port axis) rather than general 6-DoF RRT/piano-mover path planning, while `hbmartin/legolization` (`GPL-3.0-or-later`) remains strictly isolated out-of-process.
