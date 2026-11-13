# Bundled Data & Asset Licensing (`DATA_LICENSE.md`)

All first-party software source code (`src/topo_chess/`, `src/topo_brick/*.py`, `scripts/`, `tests/`, `configs/`) in `topographical-tensors` is licensed under the **MIT License** (see [`LICENSE`](LICENSE)).

To make deterministic verification (`Gates 1–8B`) self-contained without requiring external multi-gigabyte CAD downloads at runtime, `src/topo_brick/` bundles `10` compact data files (`6.29 MiB` total). Those data files are separate non-code data artifacts governed by their respective upstream licenses below (full attribution texts in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)):

| Bundled Data File | Size | Governing License | Provenance & Commercial / ShareAlike Boundary |
| :--- | ---: | :--- | :--- |
| `src/topo_brick/interface/compiled_ldraw_catalog.json.gz` | `2,210 KiB` | **`CC-BY-SA-4.0`** *(full file)* / **`CC-BY-4.0`** *(`TIER_A` + `TIER_C` subset)* | Combines derived part-hull geometry from the LDraw Official Library `2610` (`CC-BY-4.0`, `CAreadme.txt`) with `15,154` adapted `TIER_B` port-connector records derived from Roland Melkert's LDCad Shadow Library (`CC-BY-SA-4.0`). Because `TIER_B` adapts `CC-BY-SA-4.0` connector metadata, the full `.json.gz` database is licensed under **`CC-BY-SA-4.0`**. At runtime, passing `license_filter="cc_by_4_0_only"` to `TopoBrickKernel.realize()` restricts part retrieval and realization strictly to `TIER_A` + `TIER_C` (`CC-BY-4.0` only, `0` `CC-BY-SA-4.0` records). |
| `src/topo_brick/interface/compiled_mold_retrieval_index.npz` | `224 KiB` | **`CC-BY-SA-4.0`** *(full file)* / **`CC-BY-4.0`** *(`license_flags == 1` subset)* | Pre-computed 29-D volumetric/surface-normal shape descriptors over `849` mold signatures from `compiled_ldraw_catalog.json.gz`. Licensed under **`CC-BY-SA-4.0`** as a whole; the `license_flags == 1` (`CC-BY-4.0`-only) mask is enforced automatically when `license_filter="cc_by_4_0_only"`. |
| `src/topo_brick/interface/compiled_coacd_hulls.npz` | `2,949 KiB` | **`CC-BY-4.0`** | Convex polytope decompositions (`320` unique shells covering `3,273` non-convex parts) computed via `coacd` (`MIT`) strictly from LDraw Official Library `2610` part geometry (`CC-BY-4.0`). Contains zero LDCad Shadow data. |
| `src/topo_brick/interface/compiled_catalog_manifest.json` | `0.9 KiB` | **`MIT`** | First-party compiler SHA-256 provenance manifest. |
| `src/topo_brick/interface/omr_benchmark_sets.json` | `457 KiB` | **`CC-BY-4.0` / `CCAL v2.0`** | `26` human-authored Official Model Repository (`.mpd`) LEGO set models from the LDraw Official Model Repository (`0 Author:` headers preserved per model). |
| `src/topo_brick/kernel/untouched_s2b_600_cohort.json.gz` | `197 KiB` | **`MIT`** | Category-stratified `n = 600` test-split `.ldr` cohort extracted from `AvaLovelace/StableText2Brick` (Carnegie Mellon University Generative Intelligence Lab, `MIT`). |
| `src/topo_brick/evaluation/physical_validation_assets.json` | `21.5 KiB` | **`MIT`** | Physical validation assembly fixtures from CMU `StableLego` (`MIT`) and `BrickSim` (`MIT`). |
| `src/topo_brick/evaluation/raw_mesh_assets.json.gz` | `276 KiB` | **`MIT`** *(research benchmark evaluation)* | `6` continuous `.obj` example meshes (`e2, e3, e4, e5, e6, e8`) from `xjtunzy/BrickAnything` (`MIT`) plus `StableLego` (`MIT`) evaluation fixtures, included strictly for reproducible benchmark evaluation (`Gate 2`, `Gate 7A`, `Gate 8A`). |
| `src/topo_brick/evaluation/phase8_external_comparators.json` | `92.6 KiB` | **`MIT`** | First-party numerical benchmark telemetry recording out-of-process CLI evaluation metrics. |
| `src/topo_brick/kernel/untouched_s2b_600_results.json` | `13.9 KiB` | **`MIT`** | First-party numerical benchmark summary telemetry on the `n = 600` `StableText2Brick` test cohort. |

---

## Trademark Disclaimer

**LEGO® is a trademark of the LEGO Group of companies which does not sponsor, authorize, or endorse this software, dataset compilation, or research.**
