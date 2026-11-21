# Phase 11 Kernel Specification: Five Extensions to `topo_brick` for Constrained Generative & Multimodal Synthesis

**Target Package:** `topo_brick` (`src/topo_brick/`)  
**Status:** **IMPLEMENTED & DUAL-HOST VERIFIED IN `v5.2.0`** (`21 November 2026` — **All `4 / 4` Phase 11 Gates [`Gates 9A–9D`] & `113 / 113` `pytest` Tests (`topo_brick` + `topo_chess`) Passing on Both Verification Hosts with Exact `0.0000` Cross-Host Quality Identity**; archived in [`phase11-gate-verification-ledger.json`](phase11-gate-verification-ledger.json))  
**Baseline:** `v5.1.0` (`Gates 1–8B` passing across `14` `topo_brick` modules; `16,085`-part combined catalog; `0.8542` mean / `0.8769` median `5×5×6 LDU` original-mesh IoU across `80` strictly unique SHA-256 meshes; `197 / 197` standalone `independent_verifier.py` replays).  
**Scope:** Detailed technical specification of the **five kernel-level extensions** required inside `topo_brick` to support end-to-end generative brick synthesis—specifically **(1) Finite-Inventory Text-to-Brick**, **(2) Open-Catalog Text-to-Brick**, and **(3) Finite-Inventory Single-Image-to-Brick**—while preserving `topo_brick`'s foundational guarantees: **zero neural parameters in default kernel paths, strict single-core cross-host determinism, backward-compatible public APIs, and AST-isolated planner-free certificate verification.**

---

## 1. Executive Summary & Architectural Boundary

`topo_brick` `v5.1.0` solves blind 3D mesh/voxel-to-brick realization, incremental frontier legality masking, localized structural repair, and 6-DoF static equilibrium certification across `16,085` parts. Upstream perception and spatial intent planning (translating natural-language prompts into parametric 3D CSG scene graphs, or reconstructing de-lighted 3D meshes from 2D photographs) belong in an external generative orchestrator—not inside the physics kernel.

However, a code-level audit of `src/topo_brick/` against real-world finite-inventory and multimodal synthesis workloads exposes **five concrete kernel limitations** that an external wrapper cannot solve without modifying `topo_brick` itself:

| Ext # | Extension Name | Primary Target Modules | Exact `v5.1.0` Limitation Identified in Source | Certificate & Verifier Impact |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Color-Coupled Finite Inventory & Core/Skin Min-Cost Flow** | `open_catalog_realizer.py`<br>`construction_kernel.py`<br>`independent_verifier.py` | `inventory_counts: Dict[str, int]` (`open_catalog_realizer.py:1008`) tracks `part_id` (mold) only; `assign_perceptual_ldraw_colors` (`L236–L294`) assigns colors greedily assuming infinite color stock and wastes exterior colors on buried core bricks via `k_nn=8` fallback (`L280–L286`). | Schema `1.1.0`: serialize `color_id` on `placed_parts` + `colored_inventory_spec`; audit `0` `(part_id, color_id)` overruns in `InventoryCert` & `independent_verifier.py`. |
| **2** | **Inventory-Masked `uint64` Retrieval & Sparse-Kit Parity Tiling** | `catalog_retrieval_index.py`<br>`open_catalog_realizer.py`<br>`construction_kernel.py` | `retrieve_top_k` (`catalog_retrieval_index.py:813`) filters only by license tier, returning molds absent from user inventories; `open_catalog_realizer.py` disables portal fast-paths when `rem_inv` is set (`L1099`) and hardcodes `3005`/`3024` `1×1` fallbacks (`L2380–L2383`). | Emit structured `InventoryShortfallReport` alongside `PROVEN_INFEASIBLE` / `PARTIAL_OR_TIMEOUT` when sparse kits cannot span target geometry. |
| **3** | **Ribbed Hollow-Shell Target Voxelization (`HollowShellPolicy`)** | `adaptive_surface_and_macro_planner.py`<br>`open_catalog_realizer.py`<br>`phase8_round9_gates.py` | `AdaptiveSurfaceTarget.from_triangle_mesh` (`L86–L215`) ray-parity fills closed meshes `100%` solid, consuming `55%–75%` of finite brick budgets on invisible interior volume and penalizing hollow builds under solid-voxel IoU. | `GeometryCert` evaluates `shell_target_iou` against carved `evaluated_target_field`; auxiliary `exterior_crust_iou_5x5x6` audits `1`-stud surface shell fidelity. |
| **4** | **Compositional `.mpd` Subassembly Realizer & Landmark Attacher** | `construction_kernel.py`<br>`port_manifold.py`<br>`catalog_retrieval_index.py`<br>`independent_verifier.py` | `to_ldraw_mpd` (`port_manifold.py:1136`) emits flat single-block `.ldr`; `_classify_mold_and_hygiene` (`catalog_retrieval_index.py:283–345`) excludes non-flush architectural/hinged/clip molds; `export_certificate_artifact` (`independent_verifier.py:244–L255`) drops `AtomicBuildStep.subtrace`. | Serialize & replay recursive `subtrace` + rigid-group monotone corridor on `PLACE_SUBASSEMBLY`; bind `compiled_landmark_detail_index.json.gz` into `compute_catalog_sha256()`. |
| **5** | **View-Ray Visibility Weighting (`ViewPrioritySpec`)** | `open_catalog_realizer.py`<br>`adaptive_surface_and_macro_planner.py` | Global `ΔIoU` and `CIEDE2000` weight front, side, and self-occluded rear sub-voxels uniformly (`1.0`), wasting scarce slopes and colors on unseen/ambiguous back faces in single-image builds. | Unweighted 6-DoF `StaticMechanicsCert` (`TABLE_FRICTION_ONLY` / `CLUTCHED_BASEPLATE`) remains strictly enforced across all `360°`, using rear mass as counterweight. |

---

## 2. Detailed Technical Specifications

### 2.1 Extension 1 — Color-Coupled Finite Inventory (`ColoredInventorySpec`) & Two-Zone Chromatic Allocation

#### Problem Analysis
Real user part bins and retail set manifests (LDraw `.ldr` bins, Rebrickable CSVs, BrickLink XMLs) specify **joint `(part_id, color_id)` quantities**—for example, `4× (3001, color=71 Light Bluish Gray)` and `22× (3001, color=14 Yellow)`. In `v5.1.0`:
1. Passing `inventory_counts={"3001": 26}` permits placing `26` `2×4` bricks, after which `assign_perceptual_ldraw_colors` (`open_catalog_realizer.py:290`) paints all `26` bricks `color_id=71`, violating the physical inventory by `22` gray bricks.
2. Worse, for bricks completely enclosed inside thick walls or foundations (`inside` mask empty at `open_catalog_realizer.py:277`), `v5.1.0` queries the `k_nn = 8` nearest exterior mesh vertices (`L283–L286`) and paints hidden interior bricks with scarce exterior surface colors instead of consuming off-color surplus stock ("filler bricks").

#### Technical Needs & Design
1. **Canonical Data Structures & Alias Normalization (`interface/brick_catalog.py` & `kernel/construction_kernel.py`):**
   ```python
   ColoredPartKey = Tuple[str, int]  # (canonical_part_id, ldraw_color_id)

   @dataclass(frozen=True, slots=True)
   class ColoredInventorySpec:
       counts: Dict[ColoredPartKey, int]
       allow_hidden_color_substitution: bool = True
       allow_same_footprint_split: bool = True
       seam_coherence_weight: float = 0.75

       def marginal_mold_counts(self) -> Dict[str, int]:
           """Aggregate sum_c counts[(part_id, c)] per canonical mold ID for geometric solvers."""
   ```
   * Normalize incoming `part_id` tokens through `normalize_ldraw_part_id(..., COMBINED_BRICK_CATALOG)` so mold aliases (e.g., `3004p01` vs. `3004` when unpatterned fallback is enabled, or `3069a`/`3069b`) resolve deterministically without losing distinct printed IDs when explicitly requested.

2. **Exact Sub-Voxel Exterior Visibility Partition (`classify_part_surface_exposure`):**
   * Reconstruct the assembly's occupied `5×5×2 LDU` boolean grid `occ2_built` from `state.placed_parts` (and carved female cavities where open arches/windows expose interior faces).
   * For each placed brick $i \in \mathcal{P}$, inspect the 1-voxel (`5 LDU` XY / `2 LDU` Z) outward boundary shell around `pl.world_boxes`:
     * Let $A_i^{\text{ext}} \ge 0$ be the count of boundary facelets adjacent to unoccupied `~occ2_built` air (optionally weighted by Extension 5's `view_weights`).
     * Partition $\mathcal{P} = \mathcal{P}_{\text{core}} \sqcup \mathcal{P}_{\text{skin}}$ where $i \in \mathcal{P}_{\text{core}} \iff A_i^{\text{ext}} == 0$ (`100%` occluded from all exterior directions; if `primary_view_dir` is set with `strict_360_visibility=False`, rear faces remain in $\mathcal{P}_{\text{skin}}$ with reduced weight $w_{\text{back}} = 0.25$, never `0`, so `360°` inspection stays reasonable unless totally enclosed).

3. **Per-Mold Minimum-Cost Chromatic Network Flow (`allocate_inventory_colors_min_cost`):**
   * During Stage A/B geometric realization, enforce marginal mold bounds $n_m \le \sum_c N_{m,c}$ so every placed brick is guaranteed at least one physically available color in `ColoredInventorySpec`.
   * Post-geometry, for each mold $m$ with placed instances $\mathcal{P}_m$ (`|\mathcal{P}_m| \le \sum_c N_{m,c}$) and available color buckets $\mathcal{C}_m = \{c : N_{m,c} > 0\}$, solve the exact capacity-bounded linear assignment problem via `highspy` (LP relaxation is **totally unimodular** and therefore integral in $O(|\mathcal{P}_m| \cdot |\mathcal{C}_m|)$, solving in **`< 1.5 ms` total** across `500` bricks):
     $$\min_{x_{i,c} \in \{0,1\}} \sum_{i \in \mathcal{P}_m} \sum_{c \in \mathcal{C}_m} \Phi_{i,c}\, x_{i,c} \quad \text{s.t.} \quad \sum_{c \in \mathcal{C}_m} x_{i,c} = 1 \;\forall i, \quad \sum_{i \in \mathcal{P}_m} x_{i,c} \le N_{m,c} \;\forall c$$
     where the assignment cost $\Phi_{i,c}$ is:
     * **For $i \in \mathcal{P}_{\text{core}}$ (when `allow_hidden_color_substitution=True`):**
       $$\Phi_{i,c} = -\epsilon_{\text{hide}} \cdot \min_{j \in \mathcal{P}_{\text{skin}}} \Delta E_{00}\big(\text{Lab}(c),\, \mathbf{y}_j\big), \quad \epsilon_{\text{hide}} = 10^{-3}$$
       This negative tie-breaker **actively routes high-contrast "clashing" colors (e.g., bright pink, lime, yellow in a gray castle bin) into buried interior bricks first**, preserving neutral/matching colors across any cross-mold decomposition steps!
     * **For $i \in \mathcal{P}_{\text{skin}}$:**
       $$\Phi_{i,c} = \tilde{w}_i \cdot \Delta E_{00}\big(\text{Lab}(c),\, \mathbf{y}_i\big)$$
       where $\tilde{w}_i = (A_i^{\text{ext}} / A_{\max}^{\text{ext}})^{0.5} \cdot w_i^{\text{view}}$ weights larger, more prominent facade bricks higher than tiny recessed corners, followed by one deterministic Gauss-Seidel 2-swap pass (`≤ 2 ms`) reducing neighbor seam $\Delta E_{00}$ jumps (`seam_coherence_weight`).

4. **Chromatic Same-Footprint Split Pass (`allow_same_footprint_split=True`):**
   * After initial per-mold flow, inspect visible bricks $i \in \mathcal{P}_{\text{skin}}$ whose assigned color suffers $\Delta E_{00}(c_i, \mathbf{y}_i) > 12.0$ because mold $m_i$'s matching color was exhausted.
   * Test exact volume- and port-preserving sub-decompositions from *unused* `(part_id, color_id)` stock (e.g., `2×4` brick `3001` $\to$ two `2×2` `3003`s, two `1×4` `3010`s, or three stacked `2×4` `3020` plates; `1×4` plate `3710` $\to$ two `1×2` `3023`s) using `push_trial_part` / `pop_trial_part`. Accept a split iff:
     1. Mean $\Delta E_{00}$ drops by $\ge 4.0$,
     2. `check_monotone_insertion_corridor` passes for all child parts, and
     3. Incremental `can_extend_plumb_primal_witness` or `SparseStaticEquilibriumValidator.evaluate()` certifies static stability ($\eta^* \le 1.0$).

5. **Certificate & Independent Verifier Updates (`independent_verifier.py`):**
   * Increment `CERTIFICATE_SCHEMA_VERSION` to `"1.1.0"` (preserving `"1.0.0"` deserialization).
   * Add `"color_id": int(p.color_id)` to every entry in `VersionedCertificateArtifact.placed_parts` (`independent_verifier.py:220–L238`) and add `colored_inventory_spec: Optional[Dict[str, int]]` (`"part_id:color_id" -> max_qty`).
   * Extend `InventoryCert` and `verify_certificate_independently()` to verify both mold-level and `(part_id, color_id)`-level non-overrun (`0` tolerance).

---

### 2.2 Extension 2 — Inventory-Masked `uint64` Mold Retrieval & Parity-Aware Sparse-Kit Tiling

#### Problem Analysis
When a caller supplies an arbitrary LDraw parts list (e.g., an assorted `350`-piece box or dismantled vehicle set):
1. `CATALOG_RETRIEVAL_INDEX.retrieve_top_k()` (`catalog_retrieval_index.py`) searches all `426` active buildable System molds (`1,499` pose signatures across `7,178` canonical System molds) globally—returning `Top-32` slopes/wedges the user does not own while ignoring sculptural molds (`3039`, `11477`, `3659`, `85984`) the user *does* own outside a small fixed kit.
2. `open_catalog_realizer.py` (`L2380–L2383`) assumes `("3005", 1, 1)` and `("3024", 1, 1)` exist as universal `1×1` gap fillers. If a finite inventory contains zero `1×1` bricks/plates (or exhausts its `4` copies of `3005` on Course 1), odd-length `1×3` / `1×5` spans either leave structural voids that fail upper-course support or trigger inventory overruns.

#### Technical Needs & Design
1. **`O(1)` Inventory Masking in `CatalogShapeRetrievalIndex.retrieve_top_k`:**
   * Precompute `self._mold_id_to_buildable_idx: Dict[str, int]` and a reverse map from all `16,085` `COMBINED_BRICK_CATALOG` part IDs to their canonical `mold_id` (`self._part_to_canonical_mold: Dict[str, str]`, built from `_mold_equivalence_key` at load time in `< 8 ms`).
   * Add `allowed_part_ids: Optional[Collection[str]] = None` to `retrieve_top_k(...)`:
     * Map `allowed_part_ids` to canonical buildable mold indices, construct a boolean mask `inv_mask: np.ndarray [426]`, intersect `active_mold_mask = inv_mask & license_mask`, and mask `_pose_mold_idx` before vectorized `uint64` popcount scoring (`< 0.1 ms` overhead, zero `.npz` format changes).
     * When mapping selected canonical molds back to concrete `part_id`s for `AssemblyState`, rewrite each `mold_id` back to the **user's actual owned `part_id` alias** (e.g., if the user owns `3004p01` or `3040a` rather than `3040b`, place the exact owned `part_id` so `InventoryCert` and `.mpd` export match the input bin 1-to-1).

2. **Parity-Aware Course Rounding & Bounded-Inventory Strip MILP (`open_catalog_realizer.py` & `adaptive_surface_and_macro_planner.py`):**
   * **Audit Kit Footprint GCD & `1×1` Scarcity:** Before Stage-A core planning, compute remaining `1×1` single-stud capacity $S_{1\times 1} = \text{rem}(\text{"3005"}) + \lfloor \text{rem}(\text{"3024"}) / 3 \rfloor$.
   * **Scarcity Reservation & Even-Parity Morphological Snap:** When $S_{1\times 1}$ is scarce ($S_{1\times 1} < 2 \cdot nz_{\text{courses}}$), reserve `1×1` units strictly for isolated 1-stud spires/columns and apply a 2D parity-closing filter on interior horizontal `24 LDU` / `8 LDU` slices so connected row/column runs have even stud length wherever `1×2` / `2×2` / `2×4` stock is abundant.
   * **Bounded-Inventory Course Set-Packing Fallback:** Replace greedy row slicing in inventory-constrained courses with a compact per-course `highspy` binary MILP (`mip_max_leaves=32`, `< 3 ms`/course) subject to per-course fractional inventory budgets $\sum_{c \in \text{course } z} \mathbb{I}[m(c) = m] \le \tilde{N}_{m,z}$, eliminating greedy starvation of `1×2` and `2×2` corner bricks in lower courses.
   * Enable the `< 8 ms` portal-bridge & stepped-frustum fast-path (`open_catalog_realizer.py:1099`) under finite inventories by passing `rem_inv` directly into its template selector rather than gating on `rem_inv is None`.

3. **Structured `InventoryShortfallReport` Diagnostic:**
   * Add `inventory_shortfall: Optional[InventoryShortfallReport]` to `KernelRealizeResult` reporting:
     - `available_volume_ldu3` vs. `required_target_volume_ldu3`,
     - `unfilled_volume_ratio`,
     - `exhausted_critical_molds: Tuple[str, ...]` (e.g., exhausted arch/bridging beams over `forbidden_support_boxes`), and
     - `suggested_minimum_additions: Tuple[Tuple[str, int, int], ...]` (`(part_id, color_id, qty)` computed by running one unconstrained `TIER_A` slack pass and taking the set difference).

---

### 2.3 Extension 3 — Ribbed Hollow-Shell Target Voxelization (`HollowShellPolicy`) & Dual Fidelity Contract

#### Problem Analysis
`AdaptiveSurfaceTarget.from_triangle_mesh` (`adaptive_surface_and_macro_planner.py:86–215`) fills every closed mesh `100%` solid. On a `16×16×10`-brick castle keep (`~2,560` brick-equivalents solid), a `400`-brick user inventory (`Extended-1` / `Extended-3`) is forced to shrink the entire build down to a tiny `8×8×5` block where towers, gates, and crenellations collapse below the `1`-stud Nyquist limit! Real LEGO architectural sets are **hollow shells (`1`–`2` studs thick) reinforced by internal cross-walls and corbels**.

#### Technical Needs & Design
1. **`HollowShellPolicy` Specification (`kernel/adaptive_surface_and_macro_planner.py`):**
   ```python
   @dataclass(frozen=True, slots=True)
   class HollowShellPolicy:
       enabled: bool = False
       wall_thickness_studs: int = 1          # 20 LDU lateral shell thickness (min 1, or 2 for >500 pcs)
       roof_thickness_plates: int = 3         # 24 LDU top cap thickness (supports studless tiles + slopes)
       diaphragm_pitch_studs: int = 6         # Internal 1-stud shear rib every P studs (0 = no ribs)
       corbel_overhang_support: bool = True   # Auto-step inward 1 stud/course beneath flat roof spans
       preserve_grounded_collar: bool = True  # Keep 2-stud base ring at z=0 for TABLE_FRICTION_ONLY tipping margin
   ```

2. **Load-Path-Guaranteed 3D Shell & Rib Operator (`apply_hollow_shell_policy`):**
   Applied directly to `(occ_5x5x2, occ_5x5x6)` inside `AdaptiveSurfaceTarget.from_triangle_mesh(..., hollow_shell_policy=...)` in **`< 6 ms`** using separable 1D distance transforms (`scipy.ndimage.distance_transform_cdt`):
   * **Step 1 (Exterior Crust Preservation):** Retain every occupied sub-voxel within `4 * wall_thickness_studs` (`5 LDU` units) laterally of exterior air `~occ` or any `forbidden_support_boxes` opening, and within `4 * roof_thickness_plates` (`2 LDU` units) vertically of upward-exposed surfaces. This guarantees **zero change** to exterior `uint64` boundary cells, surface normals, and `Top-K` mold retrieval!
   * **Step 2 (Plumb Internal Shear Diaphragms):** Inside the carved interior void `occ_solid & ~occ_crust`, retain `1`-stud (`20 LDU`) vertical grid ribs along $x \equiv x_0 \pmod{20 P}$ and $y \equiv y_0 \pmod{20 P}$ (`P = diaphragm_pitch_studs`, anchored to target bounding-box center) wherever roof/upper mass exists directly overhead (`np.maximum.accumulate` top-down mask). This guarantees every upper roof/tower brick has a direct compressive plumb column or short `< P`-stud bridge to $z = 0$!
   * **Step 3 (45° Inverted-Slope / Stepped Corbel Ring):** Immediately beneath horizontal interior ceiling spans (`> 2` studs wide), dilate the inner wall inward by `1` stud per course downward for up to `2` courses (`corbel_overhang_support=True`), enabling standard bricks or inverted slopes (`3660`, `3665`) to close roofs **without** requiring `REMOVE_TEMP_SUPPORT` interior scaffolding that would be trapped inside a closed hollow shell! (Crucial for `InsertionCert`: trapped temporary scaffolding inside a closed roof cannot be extracted along any monotone corridor!)

3. **Dual Shell / Surface Fidelity Evaluation Contract:**
   * When `hollow_shell_policy.enabled == True`, `KernelRealizeResult.evaluated_target_field` stores the carved structural shell `SparseTiledTargetField` (so `GeometryCert.target_iou` accurately measures realization fidelity to the intended hollow structure, `~0.95+`).
   * Add `evaluate_exterior_crust_fidelity(assembly, original_solid_occ_5x5x6, crust_depth_ldu=20)` returning `exterior_crust_iou_5x5x6` (evaluating strictly on the `20 LDU` outer skin + all exterior air outside the mesh so overfill spill is penalized identically to `Gate 8A`), alongside `volume_savings_ratio` ($1 - V_{\text{shell}} / V_{\text{solid}}$).

---

### 2.4 Extension 4 — Compositional Multi-Submodel (`.mpd`) Realizer & Semantic Landmark Port Attacher

#### Problem Analysis
1. **Flat `.ldr` Output & Dropped `subtrace` Verification:** Generative architectural builds (e.g., *"castle with a single tower"*) naturally decompose into semantic modules (`courtyard`, `curtain_walls`, `gatehouse`, `single_watchtower`). Currently, `realize()` outputs a single flat `AssemblyState`, `to_ldraw_mpd()` (`port_manifold.py:1136`) writes one `0 FILE` block, and `independent_verifier.py:export_certificate_artifact` (`L244–L255`) drops `AtomicBuildStep.subtrace`, preventing planner-free replay of `PLACE_SUBASSEMBLY` traces!
2. **Unreachable Non-Flush & Functional Molds in `COMBINED_BRICK_CATALOG`:** `_classify_mold_and_hygiene` (`catalog_retrieval_index.py`) restricts `CATALOG_RETRIEVAL_INDEX`'s `426` active buildable molds strictly to flush `20 LDU` System bricks/plates/slopes/arches/tiles. Thousands of iconic parts already compiled in `COMBINED_BRICK_CATALOG`—spire cones (`4589`, `3942c`), domes (`553b`), flags with clips (`2525`, `4495`), window/door frames (`60592`, `60596`), masonry/palisade bricks (`98283`, `30136`), bars/torches (`30374`), and click-hinges (`44301`/`44302`)—are never placed by `realize()`.

#### Technical Needs & Design
1. **`SubassemblySceneGraph` & `realize_compositional_mpd` (`kernel/subassembly_realizer.py`):**
   ```python
   @dataclass(frozen=True, slots=True)
   class SubmodelNodeSpec:
       submodel_name: str                        # e.g., "single_watchtower.ldr"
       adaptive_target: AdaptiveSurfaceTarget
       world_offset_ldu: Tuple[int, int, int]    # Integer 20-LDU XY, 8-LDU Z aligned offset
       parent_submodel: Optional[str] = None
       forbidden_support_boxes: Tuple[LDUBox, ...] = ()
       hollow_shell_policy: Optional[HollowShellPolicy] = None
       semantic_style_hint: Optional[str] = None
       vertex_colors_rgb: Optional[np.ndarray] = None
       verts_ldu: Optional[np.ndarray] = None
   ```
   * **Topological Realization & Seam Stitching:**
     1. Traverse `SubmodelNodeSpec` nodes in topological parent-before-child order, sharing a single mutable `rem_inventory` (Extension 1 & 2) across the DAG.
     2. For grounded root submodels (`parent_submodel is None`), realize directly on $z = 0$ while keeping upward studs exposed (`suppress_studless_capping_boxes`) inside child attachment footprints! (Without suppressing `"smooth_studless"` tiles under a child tower's base footprint, smooth tiles would prevent the child tower from clutching onto the curtain wall!)
     3. For child submodels (`parent_submodel != None`), realize the child assembly in its local frame, verify its standalone flat-table `ExecutionTrace` (`subtrace`), verify `check_rigid_group_insertion_corridor` into the parent assembly along `+Z` (or port axis), mate interface ports into the combined `AssemblyState`, emit an `AtomicBuildStep(op_type=AtomicOpType.PLACE_SUBASSEMBLY, instance_ids=..., subtrace=...)`, and run `SparseStaticEquilibriumValidator` on the merged assembly.
   * **True Multi-Block `.mpd` Exporter (`export_compositional_mpd`):**
     - Emit `0 FILE main.ldr` containing `1 16 <x_ld> <y_ld> <z_ld> <R_ld> <submodel_name>` Type-1 submodel references (`_C_LDRAW_TO_ZUP.T` transformed), followed by each `0 FILE <submodel_name> ... 0 STEP ... 0 NOFILE` block—verified to round-trip `100%` bit-identically through `flatten_omr_mpd_to_se3(..., catalog=COMBINED_BRICK_CATALOG)`!
   * **Recursive `subtrace` Serialization & Replay in `independent_verifier.py`:**
     - Update `_serialize_atomic_step` / `_deserialize_atomic_step` in `independent_verifier.py` to preserve nested `subtrace` lists and replay both (a) each subassembly's internal step-by-step table equilibrium + `2 LDU` CSG and (b) rigid-group monotone corridor clearance + parent post-attachment 6-DoF equilibrium!

2. **`LandmarkDetailIndex` & Post-Core `SemanticLandmarkAttacher` (`interface/landmark_detail_index.py`):**
   * Compile a deterministic companion index (`compiled_landmark_detail_index.json.gz`, `< 120 KB`, SHA-256 bound into `compute_catalog_sha256()`) over non-flush architectural & functional parts in `COMBINED_BRICK_CATALOG` across `8` roles:
     `FINIAL_SPIRE_CONE`, `DOME_ROOF_CAP`, `CLIP_FLAG_BANNER`, `BAR_POLE_ANTENNA`, `WINDOW_FRAME_LATTICE`, `DOOR_PORTCULLIS`, `TEXTURED_MASONRY_BRICK`, `HINGED_PANEL_MOUNT`.
   * Expose `LandmarkAttachmentRequest(role: str, anchor_box_ldu: LDUBox, preferred_color_id: Optional[int] = None, count: int = 1, license_filter: str = "all")`:
     - Queries `state.get_active_frontier()` for `PortStatus.PLACEABLE` ports (`STANDARD_STUD`, `HOLLOW_STUD`, `BAR_3_18MM`, `CLIP_3_18MM`, `HINGE_SOCKET`/`PIN`) inside `anchor_box_ldu`.
     - Calls `kernel.constrain()` over matching `LandmarkDetailIndex` molds (filtered by `license_filter` and `rem_inventory`), verifying `2 LDU` CSG / `CoACD` `PolytopeOBBSAT` clearance, monotone insertion, and 6-DoF equilibrium before committing the attachment and appending `PLACE_PART` to `ExecutionTrace`.

---

### 2.5 Extension 5 — View-Ray Visibility Weighting (`ViewPrioritySpec`) & 2D Reprojection Scorer

#### Problem Analysis
In single-image-to-brick realization (`Extended-3`), a 3D mesh reconstructed from one photograph has high geometric/albedo confidence on the camera-facing front and silhouette rim, and high hallucination variance on the self-occluded rear. Under tight `(part_id, color_id)` budgets, `v5.1.0`'s isotropic `ΔIoU` and unweighted `CIEDE2000` spend half of the user's matching slopes and colors on the unseen back face while leaving visible errors on the front face!

#### Technical Needs & Design
1. **`ViewPrioritySpec` & Fast Sub-Voxel Ray-Caster (`compute_subvoxel_view_weights`):**
   ```python
   @dataclass(frozen=True, slots=True)
   class ViewPrioritySpec:
       view_dir_ldu: Tuple[float, float, float]  # Unit vector from camera toward object in +Z-up LDU
       camera_pos_ldu: Optional[Tuple[float, float, float]] = None  # None = orthographic
       front_visible_weight: float = 2.5
       silhouette_rim_weight: float = 3.5
       occluded_rear_weight: float = 0.5
   ```
   * Vectorized 3D DDA / depth-buffer projection over `occ_5x5x6` (`(4*nx, 4*ny, 4*nz)`, typically `≤ 80×80×40`) runs in **`< 2.5 ms`** in pure NumPy:
     1. Project occupied sub-voxel centers onto the 2D orthographic/perspective camera plane `(u, v)` at `2.5 LDU` pixel pitch and compute minimum depth $d_{\min}(u, v)$.
     2. Sub-voxels within `d <= d_min(u, v) + 6.0 LDU` are marked **camera-visible** (`front_visible_weight = 2.5`).
     3. Camera-visible sub-voxels whose `(u, v)` pixel borders background (`d_min == +inf` within a `1`-pixel 4-neighborhood) are upgraded to **silhouette-rim** (`silhouette_rim_weight = 3.5`).
     4. All remaining occupied sub-voxels receive `occluded_rear_weight = 0.5`, while **exterior empty air sub-voxels along silhouette background rays receive `spill_penalty_weight = 3.5`** so bricks never jut outside the photographed 2D silhouette!

2. **Integration Into Sculptural `ΔIoU` & Extension 1 Chromatic Flow:**
   * Precompute `cell_weight_map: np.ndarray [nx, ny, nz]` (`float32`) so Stage-B slope/wedge/arch/jumper/SNOT swaps evaluate weighted marginal `ΔIoU_w`: front and silhouette slopes win scarce mold inventory before rear slopes.
   * Multiply Extension 1's exterior chromatic weight $\tilde{w}_i$ by brick $i$'s mean ray-visibility weight $w_i^{\text{view}} \in [0.25, 3.5]$, ensuring the photographed front receives minimum-$\Delta E_{00}$ colors while the occluded rear absorbs secondary shades and acts as a **physical counterweight zone**.
   * **Uncompromised 3D Mechanics:** `SparseStaticEquilibriumValidator` (`TABLE_FRICTION_ONLY` or `CLUTCHED_BASEPLATE`) continues to evaluate exact 3D center-of-mass and 6-DoF wrenches across the *entire* assembly—guaranteeing the finished build stands stably in real physical gravity regardless of viewing angle!

3. **Fast Camera Reprojection Evaluator (`evaluate_camera_reprojection_fidelity`):**
   * Rasterizes placed `world_boxes` + `PlacedPart.color_id` (`LDRAW_SOLID_PALETTE_LAB`) onto the `(u, v)` camera plane (`2.5 LDU` pixels, `< 4 ms`) against a reference foreground mask and de-lighted target Lab image/mesh projection, returning `silhouette_iou_2d` and `visible_pixel_mean_delta_e00` in `OpenCatalogRealizeDiagnostics`.

---

## 3. Unified Public API Changes (`TopoBrickKernel.realize`)

All five extensions integrate cleanly as **optional, default-`None` parameters** on `TopoBrickKernel.realize()` (and PEP 562 lazy exports in `src/topo_brick/__init__.py`), guaranteeing **`100%` backward compatibility** across all `64` existing tests:

```python
result = kernel.realize(
    target_field,
    # Existing v5.1.0 parameters (unchanged defaults):
    solver_id="S1_CoverAndRepair",
    part_retrieval="open_catalog_top_k",
    retrieval_top_k=48,
    license_filter="all",
    semantic_style_hint=None,
    surface_refinement="open_catalog_sculpted",
    adaptive_mesh_target=adaptive_target,
    inventory_counts=None,
    # Phase 11 (Five Extensions) optional parameters:
    colored_inventory: Optional[ColoredInventorySpec] = None,          # Ext 1 & Ext 2
    hollow_shell_policy: Optional[HollowShellPolicy] = None,           # Ext 3
    subassembly_graph: Optional[Sequence[SubmodelNodeSpec]] = None,    # Ext 4A (.mpd)
    landmark_requests: Sequence[LandmarkAttachmentRequest] = (),       # Ext 4B (details)
    view_priority: Optional[ViewPrioritySpec] = None,                  # Ext 5 (2D view)
)
```

---

## 4. Recommended Implementation Order & Falsifiable Acceptance Gates (`Phase 11`)

### 4.1 Staged Build Sequence
1. **Workstream 1 (`Ext 1` + `Ext 2` — Finite Colored Inventory Foundation):** Implement `ColoredInventorySpec`, `O(1)` inventory-masked `retrieve_top_k`, parity-aware sparse-kit course MILP, core/skin min-cost chromatic flow + footprint splitting, and `VersionedCertificateArtifact` `v1.1.0` color/inventory verification.
2. **Workstream 2 (`Ext 3` — Ribbed Hollow Shells):** Implement `HollowShellPolicy` inside `AdaptiveSurfaceTarget`, corbelled roof-span support (eliminating trapped interior scaffolding), and dual `shell_target_iou` / `exterior_crust_iou_5x5x6` metrics.
3. **Workstream 3 (`Ext 5` — View-Ray Visibility Priority):** Implement `ViewPrioritySpec`, `< 2.5 ms` DDA sub-voxel visibility/silhouette weighting, weighted `ΔIoU_w` + chromatic flow integration, and `evaluate_camera_reprojection_fidelity`.
4. **Workstream 4 (`Ext 4` — `.mpd` Subassemblies & Semantic Landmarks):** Compile `compiled_landmark_detail_index.json.gz` (bound into `compute_catalog_sha256`), implement `SubassemblySceneRealizer` + `SemanticLandmarkAttacher`, and add recursive `subtrace` serialization/replay to `independent_verifier.py`.

### 4.2 Falsifiable Acceptance Gates (`Gates 9A–9D`, Single-Core `OMP=1`)

| Gate | Benchmark & Stress Protocol | Pass Criteria (`Both Verification Hosts`) |
| :--- | :--- | :--- |
| **`Gate 9A`**<br>*Colored Finite Inventory & Core Hiding (`Ext 1 & 2`)* | `n = 40` colored target meshes realized strictly under retail set `10698-1` (`790` colored bricks across `33` LDraw colors) + `n = 20` adversarial "zero-`1×1` odd-parity" bins | **`60 / 60` (`100%`) `COMPLETED_CERTIFIED` & `verify_certificate_independently` (`v1.1.0`) pass**; **`0` `(part_id, color_id)` overruns**; exterior visible `ΔE_00` reduced by **`>= 35%`** vs. random/FIFO color draw by routing clashing colors into `OCCLUDED_CORE`; `< 250 ms` `p50`. |
| **`Gate 9B`**<br>*Ribbed Hollow-Shell Scaling (`Ext 3`)* | `n = 30` medium/large architectural & statue meshes (`16×16×10` to `24×24×16` studs) paired with `HollowShellPolicy(enabled=True)` vs. solid fill | **`30 / 30` (`100%`) 6-DoF `TABLE_FRICTION_ONLY` + `InsertionCert` + `independent_verifier` pass** (zero trapped temp supports); **`>= 45%` mean brick/volume reduction** while maintaining **`exterior_crust_iou_5x5x6` within `0.015`** of solid realization. |
| **`Gate 9C`**<br>*Compositional `.mpd` & Landmarks (`Ext 4`)* | `n = 25` multi-module architectural scenes (castles with single/twin towers, arched gatehouses, spires, flags, windows, hinged gates) | **`25 / 25` (`100%`) certified** with recursive `PLACE_SUBASSEMBLY` `subtrace` replay in `independent_verifier.py`; **`100%` exact `SE(3)` `.mpd` round-trip** via `flatten_omr_mpd_to_se3`; `0` `CC-BY-SA-4.0` leaks when `license_filter="cc_by_4_0_only"`. |
| **`Gate 9D`**<br>*View-Prioritized Scarcity Study (`Ext 5`)* | Paired ablation (`ViewPrioritySpec` ON vs. OFF) across `n = 30` scarce-color single-view mesh targets (`30%` matching-color budget) | **`30 / 30` (`100%`) `TABLE_FRICTION_ONLY` certified**; camera-facing `silhouette_iou_2d` **`+0.04+` lift** and `visible_pixel_mean_delta_e00` reduced by **`>= 4.0` `ΔE_00` units** vs. isotropic baseline. |
| **Regression** | Existing `topo_brick` + `topo_chess` `pytest` suite + Phase 11 & Phase 12 contract/falsification tests + `ruff` + runtime `sys.modules` verifier isolation | **`113 / 113` tests (`87` `topo_brick` across `16` modules + `26` `topo_chess`) pass on both hosts**; `0` `ruff` warnings; `0` new mandatory third-party dependencies. |

### 4.3 Verified Dual-Host Gate Results (`v5.2.0` — [`phase11-gate-verification-ledger.json`](phase11-gate-verification-ledger.json))

| Gate | Target Pass Criteria | **Verified Dual-Host Result (`Host B` / `Host A`, `OMP_NUM_THREADS=1`)** | Status |
| :---: | :--- | :--- | :---: |
| **`Gate 9A`** | `60/60` cert & `v1.1.0` verified, `0` color overruns, `>= 35%` skin `ΔE_00` drop, `< 250 ms` `p50` | **`60 / 60` (`40/40` official Rebrickable Set `10698-1` [`790` non-spare pcs, `108` molds, `33` colors] + `20/20` zero-`1×1` odd-parity)** certified & `v1.1.0` verified (`0` overruns, `0` `1×1` emitted on odd-parity cohort); visible skin `ΔE_00` **`39.5714 -> 15.7875` (`60.1038%` reduction**, `151` core parts hidden, `202` chromatic splits); `10 / 10` tampered color mutants rejected; `InventoryShortfallReport` verified (`2,304,000 LDU³` shortfall); **`89.80 ms` B `p50`**. | **PASS** |
| **`Gate 9B`** | `30/30` `TABLE_FRICTION_ONLY` cert + `InsertionCert` + verifier pass (`0` temp supports), `>= 45%` brick/vol reduction, `<= 0.015` crust diff | **`30 / 30` (`100%`)** hollow & solid `TABLE_FRICTION_ONLY` certified, `30 / 30` continuous half-ray monotone insertion pass (`0` temp supports), **`30 / 30` `verify_certificate_independently` pass**; mean parts `482.80 -> 235.43` (**`50.8188%` brick reduction**, **`54.6512%` volume savings**); `5×5×6 LDU` exterior crust IoU `0.996936` vs. `0.996837` (**`0.000099` mean `|ΔIoU|`**, `0.002976` max). | **PASS** |
| **`Gate 9C`** | `25/25` compositional `.mpd` cert + recursive `subtrace` replay + exact `SE(3)` round-trip + `8/8` roles + `0` ShareAlike leaks | **`25 / 25` (`100%`)** multi-submodel `.mpd` scenes (`13` `"all"` + `12` `"cc_by_4_0_only"`) certified & `subtrace`-verified; **`25 / 25` exact `SE(3)` `.mpd` round-trip (`0.0 LDU` error)**; **`8 / 8` landmark roles** (`38` landmarks attached across `110`-part `LANDMARK_DETAIL_INDEX`); **`0` `CC-BY-SA-4.0` leaks**; `9 / 9` tampered `subtrace` mutants rejected (`4.91 s` B). | **PASS** |
| **`Gate 9D`** | `30/30` `TABLE_FRICTION_ONLY` cert & verified, `>= +0.04` `silhouette_iou_2d` lift, `>= 4.0` visible `ΔE_00` drop | **`30 / 30` (`100%`)** `TABLE_FRICTION_ONLY` certified & verified (`ViewPrioritySpec` ON & OFF); camera `silhouette_iou_2d` **`0.664795 -> 0.738912` (`+0.074117` lift)**; visible `ΔE_00` **`30.5180 -> 19.3786` (`-11.1394 ΔE_00` reduction**, rear occluded absorbing `47.8831`; `4.83 s` B). | **PASS** |
| **Schema & Suite** | Schema `v1.1.0` (`sha256=0e78d354...`) + `113 / 113` `pytest` + `0` `ruff` + non-editable wheel | **`113 / 113` `pytest` tests (`87` `topo_brick` across `16` modules + `26` `topo_chess`) + `0` `ruff` warnings + clean-room wheel verified on both hosts (`cross_host_metrics_exact_match = True`)**. | **PASS** |
