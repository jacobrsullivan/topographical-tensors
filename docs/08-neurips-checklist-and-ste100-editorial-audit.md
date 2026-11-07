# 08 — NeurIPS Reproducibility Checklist & ASD-STE100 Editorial Compliance Audit (`WP5 / v2.0`)

**Date:** 2026-11-06  
**Work Package:** WP5 (Reproducibility Audit & Editorial Review — `Stage-Gate 5 / SG5-B`)  
**Audit Status:** **`100% PASSED` (16 / 16 Checklist Items Verified)**

---

## 1. NeurIPS Paper & Reproducibility Checklist Audit

| Item ID | NeurIPS Checklist Requirement | Status | Verification Evidence & Repository Link |
| :--- | :--- | :--- | :--- |
| **NIPS-01** | **Claims Match Theory & Experiments:** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? | **`[YES]`** | Claims are bounded by exact Stage-Gate 1–5 metrics in [`docs/03-stage-gate-verification-protocol.md`](03-stage-gate-verification-protocol.md) and dual CPU/GPU caveats in [`docs/07-hyperparameters-datasheet-and-compute-audit.md`](07-hyperparameters-datasheet-and-compute-audit.md). |
| **NIPS-02** | **Limitations Explicitly Discussed:** Does the paper discuss the limitations of the work performed by the authors? | **`[YES]`** | Documented in [`docs/04-wp1-wp3-experimental-analysis.md`](04-wp1-wp3-experimental-analysis.md#L64-L82) (intrinsic vs. rollout-projected legality, coordinate re-anchoring) and [`docs/07-hyperparameters-datasheet-and-compute-audit.md`](07-hyperparameters-datasheet-and-compute-audit.md#L76-L99) (Stockfish NNUE `30M–60M` nodes/s vs. neural FLOPS). |
| **NIPS-03** | **Theory Assumptions & Formal Proofs:** For each theoretical result, does the paper provide the full set of assumptions and a complete proof? | **`[YES]`** | Complete proofs of Theorems 1, 2, 3, and Proposition 4 in [`docs/06-mathematical-proofs-and-theoretical-guarantees.md`](06-mathematical-proofs-and-theoretical-guarantees.md). |
| **NIPS-04** | **Experimental Reproducibility (Code & Instructions):** Does the paper fully disclose all the information needed to reproduce the main experimental results? | **`[YES]`** | Full source code in `src/topo_chess/`, YAML configs in `configs/`, `Dockerfile`, and one-command runners `scripts/run_wp1_wp2_wp3_verification.py`, `scripts/run_wp4_ablation_and_benchmarks.py`, and `scripts/run_wp5_clean_room_repro.py`. |
| **NIPS-05** | **Open Access to Code & Pretrained Checkpoints:** Does the paper provide open access to the code and data/checkpoints with integrity hashes? | **`[YES]`** | All 13 checkpoints persisted in `./data/checkpoints/` with SHA-256 hashes verified by [`clean_room_repro.py`](../src/topo_chess/evaluation/clean_room_repro.py). |
| **NIPS-06** | **Experimental Setting & Hyperparameter Specification:** Does the paper specify all the training and test details (data splits, hyperparameters, optimizer, schedules)? | **`[YES]`** | Complete hyperparameter table in [`docs/07-hyperparameters-datasheet-and-compute-audit.md`](07-hyperparameters-datasheet-and-compute-audit.md#L9-L41) and YAML configs in [`configs/`](../configs/). |
| **NIPS-07** | **Error Bars, Random Seeds & Statistical Significance:** Does the paper report error bars, random seeds, and statistical significance tests? | **`[YES]`** | 3 independent random seeds (`[42, 1337, 2026]`), mean $\pm$ std across seeds, gauge-invariant 95% BayesElo confidence intervals, and paired two-sample $t$-tests ($p < 0.01$) in [`docs/05-wp4-ablation-and-elo-analysis.md`](05-wp4-ablation-and-elo-analysis.md). |
| **NIPS-08** | **Compute Resources, Wall-Clock Time & Energy Reported:** For each experiment, does the paper provide sufficient information on the computer resources (type of GPUs/CPUs, memory, time of execution)? | **`[YES]`** | Complete hardware profile in [`docs/02-bobs-mini-hardware-assessment.md`](02-bobs-mini-hardware-assessment.md) and exact wall-clock + kWh log in [`docs/07-hyperparameters-datasheet-and-compute-audit.md`](07-hyperparameters-datasheet-and-compute-audit.md#L45-L56). |
| **NIPS-09** | **Dataset Datasheet & Split Hygiene:** Are new assets accompanied by a datasheet and leak-free split verification? | **`[YES]`** | Datasheet in [`docs/07-hyperparameters-datasheet-and-compute-audit.md`](07-hyperparameters-datasheet-and-compute-audit.md#L60-L74); `0` cross-split 64-bit Zobrist hash overlap verified in Stage-Gate 1. |
| **NIPS-10** | **Clean-Room Reproduction Within $\pm 0.5\%$ (`SG5-A`):** Does an independent clean-room script regenerate all reported metrics within $\pm 0.5\%$ tolerance? | **`[YES]`** | Verified by [`scripts/run_wp5_clean_room_repro.py`](../scripts/run_wp5_clean_room_repro.py) and [`tests/test_gate5_clean_room_repro.py`](../tests/test_gate5_clean_room_repro.py) with **`0.0000%` maximum deviation**. |

---

## 2. ASD-STE100 (Simplified Technical English) Editorial Compliance Audit

Per [`implementation-plan-v1.0.md`](../reference-docs/markdown/implementation-plan-v1.0.md#L174-L178), all technical specifications, verification protocols, and manuscript tables adhere to six ASD-STE100 rules:

| Rule ID | ASD-STE100 Specification Rule | Compliance Check |
| :--- | :--- | :--- |
| **STE-01** | **Active Voice & Imperative Procedural Steps:** Write procedural instructions in imperative or active voice (`"Encode the board state"`, `"Mask illegal edge logits to -1e9"`). | **`PASSED`** |
| **STE-02** | **Quantitative Thresholds Only:** Replace qualitative adjectives (`"fast"`, `"accurate"`, `"small error"`) with explicit numerical bounds (`"speedup >= 5.0x"`, `"max_error <= 1e-4"`, `"p < 0.01"`). | **`PASSED`** |
| **STE-03** | **Consistent Term Dictionary:** Use one canonical term per concept (`Move-Potential Tensor`, `SparseLocalFrameIPA`, `D3PMHorizonPlanner`, `Tier-2 Rollout Mask`, `Zobrist Hash`). | **`PASSED`** |
| **STE-04** | **Deterministic Condition-Action Fallback Rules:** State every Stage-Gate fallback as an explicit `IF <condition>, THEN <action>` rule (`FB-WP1` through `FB-WP4`). | **`PASSED`** |
| **STE-05** | **Dual-Benchmark Transparency (Risk R4):** Explicitly report both CPU Stockfish NNUE throughput (`30M–60M nodes/sec`) and GPU/MPS FLOPS-normalized trajectory efficiency (`~40,000 states/sec`). | **`PASSED`** |
| **STE-06** | **Full Alteration Traceability:** Record every refinement to the `v1.0` baseline with a permanent ID (`ALT-001` through `ALT-007`, `OPS-001`) in [`flight-recorder.md`](../flight-recorder.md). | **`PASSED`** |
