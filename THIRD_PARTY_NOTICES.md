# Third-Party Notices & Attributions (`THIRD_PARTY_NOTICES.md`)

This file provides copyright notices, license attributions, and clean-room boundary disclosures for third-party datasets, part libraries, and open-source software referenced or adapted in `topographical-tensors` (`topo_chess` and `topo_brick`).

---

## 1. Trademark Notice

**LEGO® is a trademark of the LEGO Group of companies which does not sponsor, authorize, or endorse this software or research.**

---

## 2. LDraw™ Official Library (`2610`) & Official Model Repository (`OMR`)

* **Affected Bundled Files:**
  * `src/topo_brick/interface/compiled_ldraw_catalog.json.gz` (derived bounding-box, bore-carved cuboid, and `TIER_C` primitive stud/tube port geometry)
  * `src/topo_brick/interface/compiled_coacd_hulls.npz` (derived convex polytope hull decompositions)
  * `src/topo_brick/interface/compiled_mold_retrieval_index.npz` (derived volumetric/normal mold descriptors)
  * `src/topo_brick/interface/omr_benchmark_sets.json` (`26` Official Model Repository `.mpd` benchmark models, retaining original `0 Author:` headers including Orion Pobursky `[OrionP]` and LDraw OMR contributors)
* **Upstream Source:** https://www.ldraw.org/ (`LDraw Official Library Release 2610` & LDraw OMR)
* **License:** Creative Commons Attribution 4.0 International License (**`CC-BY-4.0`**; legacy parts/models dual-compatible with the LDraw Creative Commons Attribution License **`CCAL v2.0`**, https://www.ldraw.org/article/249.html and `CAreadme.txt`).
* **Attribution (`CAreadme.txt`):**
  > This work uses and adapts 3D brick geometry and Official Model Repository `.mpd` models from the **LDraw™ Parts Library**, Copyright © the LDraw.org Contributors and individual part/model authors as listed in the respective LDraw part headers. LDraw™ is a trademark owned and licensed by the Estate of James Jessiman, which does not sponsor, endorse, or authorize this project. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Modifications made: rasterized bore-carved `LDUBox` colliders, `CoACD` convex hull polytopes, `29-D` mold shape descriptors, and normalized Z-up LDU coordinate frames.

---

## 3. LDCad Shadow Library (`Roland Melkert`)

* **Affected Bundled Files:**
  * `src/topo_brick/interface/compiled_ldraw_catalog.json.gz` (`15,154` `TIER_B` port-connector metadata records)
  * `src/topo_brick/interface/compiled_mold_retrieval_index.npz` (port-availability flags for `TIER_B` records)
* **Upstream Source:** Roland Melkert, *LDCad Shadow Library* (`https://github.com/RolandMelkert/LDCadShadowLibrary`, commit `9b1131fb1991f8c0bfc072325e4e12f6271aba35`)
* **License:** Creative Commons Attribution-ShareAlike 4.0 International License (**`CC-BY-SA-4.0`**, https://creativecommons.org/licenses/by-sa/4.0/).
* **Attribution & ShareAlike Notice:**
  > `TIER_B` connector-port records in `compiled_ldraw_catalog.json.gz` are adapted from the **LDCad Shadow Library** by **Roland Melkert** (http://www.melkert.net/LDCad), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Modifications made: converted `!LDCAD SNAP_CYL`, `SNAP_CLP`, `SNAP_FGR`, `SNAP_GEN`, and `SNAP_SPH` shadow annotations across LDraw subfile hierarchies into integer Z-up `LDUFrame` `PortSpec` records (`ConnectorType`, `PortPolarity`). Pursuant to Section 3(b) of `CC-BY-SA-4.0`, the adapted `compiled_ldraw_catalog.json.gz` and `compiled_mold_retrieval_index.npz` database files are distributed under **`CC-BY-SA-4.0`**. First-party Python software in `src/topo_brick/` is an independent code work licensed under the `MIT` License; users requiring strictly `CC-BY-4.0` catalog data without ShareAlike obligations may pass `license_filter="cc_by_4_0_only"` to `TopoBrickKernel.realize()` (or `cc_by_4_0_only=True` to `CatalogShapeRetrievalIndex.retrieve()`).

---

## 4. `StableText2Brick` / `BrickGPT` & `BrickWright` (`bricklayer-sdk`)

* **Affected Bundled Files:**
  * `src/topo_brick/kernel/untouched_s2b_600_cohort.json.gz` (`600` test-split `.ldr` benchmark assemblies and text captions)
* **Upstream Sources:**
  * `AvaLovelace/StableText2Brick` & `AvaLovelace1/BrickGPT` (*"Generating Physically Stable and Buildable Brick Structures from Text"*, Carnegie Mellon University Generative Intelligence Lab, https://huggingface.co/datasets/AvaLovelace/StableText2Brick)
  * `AvaLovelace1/brickwright` (`bricklayer-sdk==0.1.0`, https://github.com/AvaLovelace1/brickwright, evaluated as an external baseline in `Gate 5`)
* **License:** **MIT License**
* **Copyright & Permission Notice:**
  > Copyright (c) 2025–2026 Ava Pun / Carnegie Mellon University Generative Intelligence Lab. Distributed under the MIT License: Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files to deal in the Software without restriction, subject to inclusion of the above copyright notice and MIT permission notice.

---

## 5. `StableLego` & `BrickSim`

* **Affected Bundled Files:**
  * `src/topo_brick/evaluation/physical_validation_assets.json`
  * `src/topo_brick/evaluation/raw_mesh_assets.json.gz` (`stablelego_extra` subset)
* **Upstream Sources:**
  * `intelligent-control-lab/StableLego` (*"StableLego: Stability Analysis of Block Stacking Assembly"*, Ruixuan Liu et al., Carnegie Mellon University Intelligent Control Lab, https://github.com/intelligent-control-lab/StableLego)
  * `intelligent-control-lab/BrickSim` (Carnegie Mellon University Intelligent Control Lab, https://github.com/intelligent-control-lab/BrickSim)
* **License:** **MIT License**
* **Copyright & Permission Notice:**
  > Copyright (c) 2024–2026 Carnegie Mellon University Intelligent Control Lab. Distributed under the MIT License.

---

## 6. `BrickAnything` Example Benchmark Meshes

* **Affected Bundled Files:**
  * `src/topo_brick/evaluation/raw_mesh_assets.json.gz` (`brickanything_meshes`: `e2.obj`, `e3.obj`, `e4.obj`, `e5.obj`, `e6.obj`, `e8.obj`)
* **Upstream Source:** `xjtunzy/BrickAnything` (*"BrickAnything: Geometry-Conditioned Buildable Brick Generation with Structure-Aware Tokenization"*, Xi'an Jiaotong University, https://github.com/xjtunzy/BrickAnything)
* **License:** **MIT License** (`https://github.com/xjtunzy/BrickAnything/blob/main/LICENSE`)
* **Notice:**
  > Copyright (c) 2025–2026 BrickAnything Authors. Repository licensed under the MIT License. The `6` example `.obj` triangle meshes in `raw_mesh_assets.json.gz` are included strictly for reproducible scientific benchmark evaluation (`Gate 2`, `Gate 7A`, `Gate 8A`); downstream users should verify underlying 3D asset provenance prior to any non-benchmark commercial reuse.

---

## 7. Out-of-Process External Benchmark Comparator Disclosure (`hbmartin/legolization`)

* **Upstream Source:** Harold Martin, `hbmartin/legolization` (`legolization==0.5.0` PyPI CLI release, https://github.com/hbmartin/legolization), licensed under **`GPL-3.0-or-later`**.
* **Clean-Room Isolation:** **Zero lines** of `legolization` source code or binaries are imported, linked, subclassed, or distributed in `topographical-tensors` (enforced by automated AST inspection in `tests/test_topo_brick/test_r9_gates1_to_5_phase8_verification.py`). When optional live comparison is requested via `--legolization-bin`, `legolization` is invoked strictly as an external out-of-process CLI executable over `.stl` / `.ldr` files; `src/topo_brick/evaluation/phase8_external_comparators.json` archives only first-party numerical timing and geometry telemetry.
