"""
Phase 14 (`v5.2.1`) Upstream Kernel Audit & Regression Suite (`TB-FIX-01` through `TB-FIX-05`):

1. `TB-FIX-01`: Calibrated per-part `.dat` SE(3) export/import frames across all `213` curated `TIER_A`
   parts + `16,085`-part audit ledger (`docs/ldraw-dat-frame-audit-v5.2.1.json`, `0` genuine frame errors,
   `58176` bore-carved native cavity, `3957b` upper-wing `0.0 LDU` seating, rotated asymmetric molds across
   `0°/90°/180°/270°`, and 3-level nested non-identity rotated `.mpd` round-trip & certification).
2. `TB-FIX-02`: Shared wall-clock LP deadlines across BOTH `highspy` and `scipy.optimize.linprog`
   (`KernelResultStatus.BUDGET_EXCEEDED`, `<= 250 ms` grace) + cantilevered tabletop biplane macro-block
   pre-screening respecting `forbidden_support_boxes` and preserving interplane wing gaps.
3. `TB-FIX-03`: Complete `322`-entry official `LDConfig.ldr` palette (`57` `Trans_Orange` `#F08F1C`,
   `ALPHA 128`), `0` fallback-to-gray misses across all `322` `LDConfig` colors and `33` Set `10698-1`
   colors, and finish-aware `ColoredInventorySpec` substitution penalties.
4. `TB-FIX-04`: Upright `+Z` orientation for `BAR_POLE_ANTENNA` on upward ports, deterministic XY-centroid
   stud tie-breaking on odd- and even-width wings, and non-upright lateral attachment preservation.
5. `TB-FIX-05`: Explicit `ground_plane_z_ldu` nonzero ground-plane datum (`-24.0 LDU` / `LDraw Y = 0`),
   pose-preserving `.ldr` round-trip, certificate schema `"1.1.1"`, backward-compatible `"1.1.0"` replay,
   and fail-closed rejection of forged `ground_plane_z_ldu` under default `VerificationObligations()`.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pytest

import topo_brick
from topo_brick.evaluation.independent_verifier import (
    CERTIFICATE_SCHEMA_VERSION,
    VerificationObligations,
    VersionedCertificateArtifact,
    export_certificate_artifact,
    main as verifier_cli_main,
    verify_certificate_independently,
)
from topo_brick.evaluation.phase9_compiler_and_hierarchical_gates import COMBINED_BRICK_CATALOG
from topo_brick.evaluation.static_equilibrium import (
    FoundationPolicy,
    SparseStaticEquilibriumValidator,
)
from topo_brick.interface.brick_catalog import (
    ColoredInventorySpec,
    SET_10698_COLORED_INVENTORY,
    get_brick_primitive,
)
from topo_brick.interface.ldraw_palette import (
    LDRAW_COLOR_BY_CODE,
    LDRAW_FULL_PALETTE_RECORDS,
    get_ldraw_color_finish,
    resolve_ldraw_color_record,
)
from topo_brick.interface.ldu_frames import C4_Z_ROT_IDS, LDUBox, LDUFrame, rot_id_to_matrix
from topo_brick.interface.port_manifold import AssemblyState
from topo_brick.interface.release_provenance import (
    PINNED_RELEASE_CATALOG_SHA256_V110,
    PINNED_RELEASE_CATALOG_SHA256_V111,
)
from topo_brick.interface.se3_kinematics import (
    ContinuousAssemblyState,
    ContinuousSE3Pose,
    _topo_brick_se3_to_ldraw_line,
    extract_se3_port_connections,
    flatten_omr_mpd_to_se3,
)
from topo_brick.kernel.colored_inventory_allocator import (
    _palette_maps,
    allocate_inventory_colors_min_cost,
)
from topo_brick.kernel.construction_kernel import (
    AtomicBuildStep,
    AtomicOpType,
    ExecutionTrace,
    KernelResultStatus,
    TopoBrickKernel,
)
from topo_brick.kernel.subassembly_realizer import (
    LandmarkAttachmentRequest,
    SubmodelRealizationRecord,
    attach_semantic_landmarks,
    export_compositional_mpd,
)
from topo_brick.representation.interface_fields import SparseTiledTargetField

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tb_fix_01_calibrated_dat_frames_and_audit_ledger() -> None:
    """Verify `TB-FIX-01` calibrated `.dat` SE(3) frames across all 3 mandatory fixtures + `16,085`-part ledger."""
    # 0. Identity reproducer closure for 3957b.dat (bottom ring at y_local = +8.0 -> t_0_ld = (0, -8, 0), R_0_ld = I)
    ident = ContinuousSE3Pose.from_matrix_and_pos(np.eye(3), (0.0, 0.0, 0.0), orthonormalize=False)
    line_id = _topo_brick_se3_to_ldraw_line("3957b", ident, 15)
    assert line_id == "1 15 0 -8 0 1 0 0 0 1 0 0 0 1 3957b.dat"

    # 1. Fixture 1: Assembled airplane upper-wing seating at z = 120 (LDraw Y_deck = -120.0)
    wing_state = AssemblyState()
    for z_col in range(0, 120, 24):
        wing_state.place_root_part("3003", LDUFrame(0, (120, 100, z_col)), is_grounded=(z_col == 0))
    lm_attached = attach_semantic_landmarks(
        kernel=TopoBrickKernel(),
        state=wing_state,
        requests=(
            LandmarkAttachmentRequest(
                role="BAR_POLE_ANTENNA",
                anchor_box_ldu=LDUBox(110, 90, 118, 150, 130, 122),
                preferred_color_id=4,
            ),
        ),
        allowed_part_ids=("3957b",),
    )
    assert len(lm_attached) == 1
    ant_iid, ant_pid, ant_pose, _ = lm_attached[0]
    assert ant_pid == "3957b"
    assert ant_pose.z_axis == (0, 0, 1)

    ldr_wing = wing_state.to_ldraw_mpd()
    ant_lines = [ln.strip() for ln in ldr_wing.splitlines() if ln.strip().endswith("3957b.dat")]
    assert len(ant_lines) == 1
    toks_ant = ant_lines[0].split()
    y_ld_ant = float(toks_ant[3])
    # Official 3957b.dat physical bottom rim is at local y = +8.0; wing deck top is at LDraw Y = -120.0
    world_bottom_rim_y = y_ld_ant + 8.0
    assert abs((-120.0) - world_bottom_rim_y) <= 0.5
    assert abs((-120.0) - world_bottom_rim_y) == pytest.approx(0.0, abs=1e-6)

    reimp_wing = ContinuousAssemblyState.from_ldr_text(ldr_wing)
    assert reimp_wing.num_parts == wing_state.num_parts
    flat_wing = flatten_omr_mpd_to_se3(ldr_wing, catalog=COMBINED_BRICK_CATALOG)
    compat_w, _, _ = extract_se3_port_connections(flat_wing, pos_tol_ldu=0.5)
    assert compat_w >= 5
    ant_cp = [cp for cp in reimp_wing.placed_parts.values() if cp.part_id == "3957b"][0]
    assert float(np.dot(ant_cp.continuous_pose.R[:, 2], np.array([0.0, 0.0, 1.0]))) == pytest.approx(1.0, abs=1e-6)
    assert any(ant_iid in (ga[0], gb[0]) for ga, gb in reimp_wing.matched_port_pairs.items())

    # 2. Fixture 2: Rotated asymmetric molds across all 4 cardinal yaw angles (0, 90, 180, 270 deg)
    for asym_pid in ("3039", "4286", "2357", "60621"):
        prim_a = get_brick_primitive(asym_pid)
        seen_x_dirs = set()
        for rid_yaw in C4_Z_ROT_IDS:
            st_asym = AssemblyState()
            if asym_pid == "60621":
                st_asym.place_root_part("60596", LDUFrame(rid_yaw, (40, 40, 0)), is_grounded=True)
                pool_door = st_asym.generate_candidate_pool(allowed_part_ids=["60621"])
                assert len(pool_door.candidates) >= 1
                cand_door = pool_door.candidates[0]
                pose_a = cand_door.pose
                st_asym.commit_candidate(cand_door)
            else:
                st_asym.place_root_part("3031", LDUFrame(0, (40, 40, 0)), is_grounded=True)
                pose_a = LDUFrame(rid_yaw, (40, 40, 8 - min(b.zmin for b in prim_a.hull_boxes)))
                st_asym.place_root_part(asym_pid, pose_a, is_grounded=False)

            ldr_a = st_asym.to_ldraw_mpd()
            se3_a = ContinuousAssemblyState.from_ldr_text(ldr_a)
            assert se3_a.num_parts == 2
            imp_asym = [cp for cp in se3_a.placed_parts.values() if cp.part_id == asym_pid][0]
            exp_R = rot_id_to_matrix(pose_a.rot_id).astype(np.float64)
            exp_t = np.array(pose_a.pos, dtype=np.float64)
            assert np.allclose(imp_asym.continuous_pose.R, exp_R, atol=1e-5)
            assert np.allclose(imp_asym.continuous_pose.t, exp_t, atol=1e-4)
            seen_x_dirs.add(tuple(int(round(v)) for v in imp_asym.continuous_pose.R[:, 0]))
            compat_a, _, _ = extract_se3_port_connections(
                flatten_omr_mpd_to_se3(ldr_a, catalog=COMBINED_BRICK_CATALOG), pos_tol_ldu=0.5
            )
            assert compat_a >= 1
        assert len(seen_x_dirs) == 4, f"Expected 4 distinct world yaw orientations for {asym_pid}"

    # 3. Fixture 3: 3-level nested .mpd with non-identity offsets & rotations (`world_rot_id = 9` and `10`)
    pod_local = AssemblyState()
    pod_local.place_root_part("3023", LDUFrame(0, (0, 0, 0)), is_grounded=True)  # 1x2 plate z=[0..8]
    pod_local.place_root_part("3039", LDUFrame(0, (0, 0, 8)), is_grounded=False)  # 2x2 slope z=[8..32]
    pod_local.place_root_part("3957b", LDUFrame(0, (0, 0, 32)), is_grounded=False)  # antenna on rear stud

    wing_tf = LDUFrame(9, (60, 20, 24))
    pod_tf = LDUFrame(10, (20, 60, 32))

    world_mpd_state = AssemblyState()
    i_root = world_mpd_state.place_root_part("3001", LDUFrame(0, (40, 20, 0)), is_grounded=True)
    i_wing = world_mpd_state.place_root_part(
        "3031", wing_tf.compose(LDUFrame(0, (20, 0, 0))), is_grounded=False
    )
    pod_entries = []
    for lp_s in pod_local.placed_parts.values():
        w_iid_p = world_mpd_state.place_root_part(lp_s.part_id, pod_tf.compose(lp_s.pose), is_grounded=False)
        pod_entries.append((w_iid_p, lp_s.part_id, lp_s.pose, 15))

    records = (
        SubmodelRealizationRecord(
            submodel_name="wing_assy.ldr",
            world_offset_ldu=wing_tf.pos,
            local_part_entries=((i_wing, "3031", LDUFrame(0, (20, 0, 0)), 15),),
            world_rot_id=wing_tf.rot_id,
            parent_submodel=None,
        ),
        SubmodelRealizationRecord(
            submodel_name="tip_pod.ldr",
            world_offset_ldu=pod_tf.pos,
            local_part_entries=tuple(pod_entries),
            world_rot_id=pod_tf.rot_id,
            parent_submodel="wing_assy.ldr",
        ),
    )
    mpd_text = export_compositional_mpd(
        submodel_records=records,
        state=world_mpd_state,
        extra_root_instance_ids=(i_root,),
    )
    blocks = mpd_text.split("0 FILE ")
    wing_block = [b for b in blocks if b.startswith("wing_assy.ldr")][0]
    assert "tip_pod.ldr" in wing_block

    flat_placements = flatten_omr_mpd_to_se3(mpd_text, catalog=COMBINED_BRICK_CATALOG)
    assert len(flat_placements) == 5
    exp_slope_pose = pod_tf.compose(LDUFrame(0, (0, 0, 8)))
    imp_slope = [fp for fp in flat_placements if fp.canonical_part_id == "3039"][0]
    assert imp_slope.continuous_pose is not None
    assert np.allclose(imp_slope.continuous_pose.R, rot_id_to_matrix(exp_slope_pose.rot_id), atol=1e-5)
    assert np.allclose(imp_slope.continuous_pose.t, np.array(exp_slope_pose.pos, dtype=np.float64), atol=1e-4)

    # Also verify rotated `PLACE_SUBASSEMBLY` passes BOTH `TopoBrickKernel.certify()` and `independent_verifier`!
    world_combined = AssemblyState()
    p0 = world_combined.place_root_part("3001", LDUFrame(0, (0, 20, 0)), is_grounded=True)
    sub_iids = []
    sub_steps = []
    sub_tf = LDUFrame(9, (20, 20, 24))
    for idx_s, lp_s in enumerate(pod_local.placed_parts.values()):
        wp_fr = sub_tf.compose(lp_s.pose)
        w_iid = world_combined.place_root_part(lp_s.part_id, wp_fr, is_grounded=False)
        sub_iids.append(w_iid)
        sub_steps.append(
            AtomicBuildStep(
                op_type=AtomicOpType.PLACE_PART,
                instance_ids=(w_iid,),
                part_id=lp_s.part_id,
                pose=lp_s.pose,
                is_grounded=(idx_s == 0),
            )
        )
    rot_trace = ExecutionTrace(
        steps=(
            AtomicBuildStep(
                op_type=AtomicOpType.PLACE_PART,
                instance_ids=(p0,),
                part_id="3001",
                pose=LDUFrame(0, (0, 20, 0)),
                is_grounded=True,
            ),
            AtomicBuildStep(
                op_type=AtomicOpType.PLACE_SUBASSEMBLY,
                instance_ids=tuple(sub_iids),
                part_id="tip_pod",
                pose=sub_tf,
                subtrace=tuple(sub_steps),
            ),
        ),
        allow_subassemblies=True,
    )
    kernel = TopoBrickKernel()
    certs_sub = kernel.certify(world_combined, execution_trace=rot_trace, foundation_policy="clutched_baseplate")
    assert certs_sub.overall_certified is True
    art_sub = export_certificate_artifact(world_combined, certs_sub, validator=kernel.validator)
    rep_sub = verify_certificate_independently(art_sub)
    assert rep_sub.verified is True, f"Rotated subassembly replay failed: {rep_sub.failure_reasons}"

    # 4. Verify committed zero-PII 16,085-part audit report (`docs/ldraw-dat-frame-audit-v5.2.1.json`)
    audit_path = REPO_ROOT / "docs" / "ldraw-dat-frame-audit-v5.2.1.json"
    assert audit_path.is_file()
    raw_txt = audit_path.read_text(encoding="utf-8")
    assert ("/" + "Users" + "/") not in raw_txt
    assert "/home/" not in raw_txt
    assert ("/" + "private" + "/" + "tmp") not in raw_txt
    audit_d = json.loads(raw_txt)
    assert audit_d["kernel_version"] == "5.2.1"
    assert audit_d["total_catalog_ids_audited"] == 16085
    assert audit_d["tier_a_curated_213"]["resolved_dat_count"] == 213
    assert audit_d["total_genuine_frame_errors"] == 0
    assert (
        audit_d["spotlight_verifications"]["3957b_upper_wing_z120_seating"][
            "signed_vertical_separation_ldu"
        ]
        == 0.0
    )
    assert (
        audit_d["spotlight_verifications"]["58176_native_frame_bore_carving"][
            "is_ldraw_native_frame"
        ]
        is True
    )


def test_tb_fix_02_shared_deadlines_highs_and_scipy_and_biplane_supports() -> None:
    """Verify `TB-FIX-02` shared deadlines on both HiGHS & SciPy + cantilevered biplane supports & wing gaps."""
    from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
    from topo_brick.interface.ldraw_shadow_catalog_compiler import COMPILED_LDRAW_CATALOG

    COMPILED_LDRAW_CATALOG._ensure_loaded()
    CATALOG_RETRIEVAL_INDEX._ensure_loaded()

    # 1. Direct validator deadline enforcement on BOTH native HiGHS and forced SciPy fallback
    st_2 = AssemblyState()
    st_2.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_2.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)

    for force_scipy in (False, True):
        val = SparseStaticEquilibriumValidator(_force_scipy_backend=force_scipy)
        eq_ok = val.evaluate(st_2, foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY)
        assert eq_ok.is_stable is True
        assert eq_ok.budget_exceeded is False

        expired_dl = time.monotonic() - 0.005
        t_s = time.perf_counter()
        eq_exp = val.evaluate(
            st_2,
            foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
            deadline_monotonic_s=expired_dl,
        )
        assert (time.perf_counter() - t_s) <= 0.05
        assert eq_exp.is_stable is False
        assert eq_exp.budget_exceeded is True

    # 2. End-to-end `TopoBrickKernel.realize()` budget enforcement & <= 250 ms grace across HiGHS and SciPy
    big_airplane_boxes = (
        LDUBox(190, 70, 0, 350, 470, 96),
        LDUBox(190, 30, 96, 350, 510, 288),
        LDUBox(-10, 110, 96, 550, 270, 192),
        LDUBox(30, 150, 192, 110, 230, 384),
        LDUBox(430, 150, 192, 510, 230, 384),
        LDUBox(-10, 110, 384, 550, 270, 480),
        LDUBox(230, 390, 288, 310, 510, 576),
    )
    tf_big = SparseTiledTargetField(boxes=big_airplane_boxes)
    kernel_warm = TopoBrickKernel()
    _ = kernel_warm.realize(
        target_field=tf_big,
        part_retrieval="open_catalog_top_k",
        retrieval_top_k=16,
        foundation_policy="table_friction_only",
        max_wall_clock_s=0.02,
    )

    for force_scipy in (False, True):
        k_dl = TopoBrickKernel(validator=SparseStaticEquilibriumValidator(_force_scipy_backend=force_scipy))
        t0 = time.perf_counter()
        res_dl = k_dl.realize(
            target_field=tf_big,
            part_retrieval="open_catalog_top_k",
            retrieval_top_k=24,
            foundation_policy="table_friction_only",
            max_wall_clock_s=0.20,
        )
        dt = time.perf_counter() - t0
        assert res_dl.status == KernelResultStatus.BUDGET_EXCEEDED
        assert res_dl.stopping_reason == "wall_clock_deadline_exceeded"
        assert res_dl.remaining_budget_s == 0.0
        assert dt <= 0.45, f"Deadline grace exceeded 250 ms (force_scipy={force_scipy}): dt={dt:.3f}s"

    # 3. Cantilevered tabletop biplane (`macro_block_mode="force"`, `table_friction_only`, `minimal_permanent_pillars`)
    #    preserving interplane wing windows via `forbidden_support_boxes`!
    biplane_boxes = (
        LDUBox(120, 40, 0, 200, 160, 24),     # narrow central table skid
        LDUBox(100, 0, 24, 220, 200, 72),     # fuselage body (x = 100..220)
        LDUBox(100, 60, 72, 220, 140, 96),    # center cabane / cockpit deck (x = 100..220)
        LDUBox(0, 60, 24, 320, 140, 48),      # cantilevered lower wing (z = 24..48)
        LDUBox(20, 60, 48, 60, 140, 96),      # left outer interplane cabane strut (x = 20..60)
        LDUBox(260, 60, 48, 300, 140, 96),    # right outer interplane cabane strut (x = 260..300)
        LDUBox(0, 60, 96, 320, 140, 120),     # upper biplane wing (z = 96..120)
    )
    interplane_gap_boxes = (
        LDUBox(60, 60, 48, 100, 140, 96),
        LDUBox(220, 60, 48, 260, 140, 96),
    )
    tf_biplane = SparseTiledTargetField(boxes=biplane_boxes)
    kernel = TopoBrickKernel()
    res_bp = kernel.realize(
        target_field=tf_biplane,
        macro_block_mode="force",
        foundation_policy="table_friction_only",
        support_policy="minimal_permanent_pillars",
        forbidden_support_boxes=interplane_gap_boxes,
    )
    assert res_bp.status == KernelResultStatus.COMPLETED_CERTIFIED
    assert res_bp.certificates.overall_certified is True
    for pl in res_bp.assembly.placed_parts.values():
        for wb in pl.world_boxes:
            for gap_b in interplane_gap_boxes:
                assert not wb.intersects(gap_b), f"Placed part {pl.part_id} at {wb} violated interplane wing gap {gap_b}"

    art_bp = export_certificate_artifact(
        res_bp.assembly,
        res_bp.certificates,
        validator=kernel.validator,
        target_field=tf_biplane,
        support_policy="minimal_permanent_pillars",
        forbidden_support_boxes=interplane_gap_boxes,
    )
    ver_bp = verify_certificate_independently(
        art_bp,
        VerificationObligations(expected_forbidden_boxes=interplane_gap_boxes),
    )
    assert ver_bp.verified is True, f"Biplane independent verification failed: {ver_bp.failure_reasons}"


def test_tb_fix_03_full_322_ldconfig_palette_set10698_and_finish_penalties() -> None:
    """Verify `TB-FIX-03` 322-color `LDConfig.ldr` lookup, Set `10698-1` color `57`, and finish penalties."""
    assert topo_brick.__version__ == "5.2.1"
    assert len(LDRAW_FULL_PALETTE_RECORDS) == 322
    assert len(LDRAW_COLOR_BY_CODE) == 322

    c57 = LDRAW_COLOR_BY_CODE[57]
    assert c57.name == "Trans_Orange"
    assert c57.hex_rgb.upper() == "#F08F1C"
    assert c57.alpha == 128
    assert c57.finish == "TRANSPARENT"
    assert resolve_ldraw_color_record(182) == c57
    assert get_ldraw_color_finish(297) == "PEARLESCENT"
    assert get_ldraw_color_finish(80) == "METALLIC"

    # Verify `_palette_maps()` covers all 322 official LDConfig colors + all 33 Set 10698-1 colors with 0 misses
    cid_to_lab, _, _ = _palette_maps()
    for cid in LDRAW_COLOR_BY_CODE:
        assert cid in cid_to_lab

    set_colors = set(SET_10698_COLORED_INVENTORY.distinct_colors())
    assert len(set_colors) == 33
    assert 57 in set_colors
    assert 182 not in set_colors
    for cid in set_colors:
        assert cid in LDRAW_COLOR_BY_CODE

    # Verify finish-aware substitution penalties prevent transparent `57` from replacing matte orange `25` by default
    st = AssemblyState()
    iid0 = st.place_root_part("3024", LDUFrame(0, (0, 0, 0)), is_grounded=True, color_id=25)  # Orange (#FE8A18)
    verts_orange = np.array([[0.0, 0.0, 4.0]], dtype=np.float64)
    cols_orange = np.array([[0xFE / 255.0, 0x8A / 255.0, 0x18 / 255.0]], dtype=np.float64)

    inv_matte = ColoredInventorySpec(
        counts={
            ("3024", 57): 1,   # Trans_Orange (#F08F1C, very close RGB, TRANSPARENT)
            ("3024", 191): 1,  # Bright_Light_Orange (#F8BB3D, farther RGB, SOLID)
        },
        enable_translucency=False,
        trans_substitution_penalty_de00=18.0,
    )
    allocate_inventory_colors_min_cost(
        st, inv_matte, verts_ldu=verts_orange, vertex_colors_rgb=cols_orange
    )
    assert st.placed_parts[iid0].color_id == 191

    st_tr = AssemblyState()
    iid1 = st_tr.place_root_part("3024", LDUFrame(0, (0, 0, 0)), is_grounded=True, color_id=25)
    inv_trans_ok = ColoredInventorySpec(
        counts={
            ("3024", 57): 1,
            ("3024", 191): 1,
        },
        enable_translucency=True,
    )
    allocate_inventory_colors_min_cost(
        st_tr, inv_trans_ok, verts_ldu=verts_orange, vertex_colors_rgb=cols_orange
    )
    assert st_tr.placed_parts[iid1].color_id == 57


def test_tb_fix_04_upright_antenna_and_deterministic_wing_centroid_tiebreak() -> None:
    """Verify `TB-FIX-04` upright `+Z` antenna on odd/even wings + deterministic tie-break + lateral role."""
    kernel = TopoBrickKernel()

    # 1. Odd-width upper wing (13 studs wide in X: x in [0..260], exact stud at center x = 130)
    odd_wing = AssemblyState()
    for k in range(6):
        odd_wing.place_root_part("3023", LDUFrame(9, (10 + 40 * k, 10, 0)), is_grounded=True)
    odd_wing.place_root_part("3024", LDUFrame(0, (250, 10, 0)), is_grounded=True)
    att_odd = attach_semantic_landmarks(
        kernel=kernel,
        state=odd_wing,
        requests=(
            LandmarkAttachmentRequest(
                role="BAR_POLE_ANTENNA",
                anchor_box_ldu=LDUBox(0, 0, 6, 260, 20, 10),  # exact XY center = (130.0, 10.0)
            ),
        ),
        allowed_part_ids=("3957b",),
    )
    assert len(att_odd) == 1
    assert att_odd[0][2].pos == (130, 10, 8)
    assert att_odd[0][2].z_axis == (0, 0, 1)

    # 2. Even-width upper wing (14 studs wide in X: x in [0..280], exact center x = 140 between studs 130 & 150)
    poses_even = []
    for reverse_order in (False, True):
        even_wing = AssemblyState()
        xs_plates = list(range(7))
        if reverse_order:
            xs_plates.reverse()
        for k in xs_plates:
            even_wing.place_root_part("3023", LDUFrame(9, (10 + 40 * k, 10, 0)), is_grounded=True)
        att_ev = attach_semantic_landmarks(
            kernel=kernel,
            state=even_wing,
            requests=(
                LandmarkAttachmentRequest(
                    role="BAR_POLE_ANTENNA",
                    anchor_box_ldu=LDUBox(0, 0, 6, 280, 20, 10),  # exact XY center = (140.0, 10.0)
                ),
            ),
            allowed_part_ids=("3957b",),
        )
        assert len(att_ev) == 1
        assert att_ev[0][2].z_axis == (0, 0, 1)
        assert att_ev[0][2].pos[0] in (130, 150)
        poses_even.append(att_ev[0][2])
    assert poses_even[0] == poses_even[1]
    assert poses_even[0].pos == (130, 10, 8)

    # 3. Non-upright lateral SNOT attachment regression check (both non-antenna and lateral antenna ports)
    snot_state = AssemblyState()
    snot_state.place_root_part("87087", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    snot_state.place_root_part("3070b", LDUFrame(0, (0, 0, 24)), is_grounded=False)  # cap top stud so only lateral SNOT stud is open
    att_lat = attach_semantic_landmarks(
        kernel=kernel,
        state=snot_state,
        requests=(
            LandmarkAttachmentRequest(
                role="CLIP_FLAG_BANNER",
                anchor_box_ldu=LDUBox(-20, -30, 0, 20, 30, 24),
            ),
        ),
        allowed_part_ids=("2555", "11476"),
    )
    assert len(att_lat) == 1
    assert att_lat[0][2].z_axis[2] == 0, f"Expected horizontal SNOT attachment, got z_axis={att_lat[0][2].z_axis}"

    snot_ant_state = AssemblyState()
    snot_ant_state.place_root_part("87087", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    snot_ant_state.place_root_part("3070b", LDUFrame(0, (0, 0, 24)), is_grounded=False)  # cap top stud so only lateral SNOT stud is open
    att_lat_ant = attach_semantic_landmarks(
        kernel=kernel,
        state=snot_ant_state,
        requests=(
            LandmarkAttachmentRequest(
                role="BAR_POLE_ANTENNA",
                anchor_box_ldu=LDUBox(-20, -30, 0, 20, 30, 24),
            ),
        ),
        allowed_part_ids=("3957b",),
    )
    assert len(att_lat_ant) == 1
    assert att_lat_ant[0][2].z_axis[2] == 0, f"Expected lateral antenna on lateral SNOT stud, got {att_lat_ant[0][2].z_axis}"


def test_tb_fix_05_nonzero_ground_plane_datum_and_schema_111_compat(tmp_path: Path) -> None:
    """Verify `TB-FIX-05` nonzero `ground_plane_z_ldu = -24.0`, Schema `"1.1.1"`, and `"1.1.0"` compatibility."""
    assert CERTIFICATE_SCHEMA_VERSION == "1.1.1"
    kernel = TopoBrickKernel()

    # 1. Nonzero ground datum z = -24.0 (`LDraw Y = 0`)
    st_nz = AssemblyState()
    st_nz.place_root_part("3001", LDUFrame(0, (0, 0, -24)), is_grounded=True)
    st_nz.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=False)

    certs_wrong_datum = kernel.certify(st_nz, foundation_policy="table_friction_only", ground_plane_z_ldu=0.0)
    assert certs_wrong_datum.overall_certified is False

    certs_nz = kernel.certify(st_nz, foundation_policy="table_friction_only", ground_plane_z_ldu=-24.0)
    assert certs_nz.overall_certified is True
    assert certs_nz.ground_plane_z_ldu == -24.0

    # Pose-preserving .ldr export -> ContinuousAssemblyState import -> re-export (`0.000 LDU` drift!)
    ldr_1 = st_nz.to_ldraw_mpd()
    se3_nz = ContinuousAssemblyState.from_ldr_text(ldr_1, ground_plane_z_ldu=-24.0)
    assert sum(1 for p in se3_nz.placed_parts.values() if p.is_grounded) == 1
    eq_se3_nz = se3_nz.evaluate_static_equilibrium(
        foundation_policy=FoundationPolicy.TABLE_FRICTION_ONLY,
        ground_plane_z_ldu=-24.0,
    )
    assert eq_se3_nz.is_stable is True
    ldr_2 = se3_nz.to_ldr_text()
    se3_nz_2 = ContinuousAssemblyState.from_ldr_text(ldr_2, ground_plane_z_ldu=-24.0)
    for iid in sorted(st_nz.placed_parts):
        orig_pos = np.array(st_nz.placed_parts[iid].pose.pos, dtype=np.float64)
        imp_pos = se3_nz.placed_parts[iid].continuous_pose.t
        re_pos = se3_nz_2.placed_parts[iid].continuous_pose.t
        assert np.allclose(orig_pos, imp_pos, atol=1e-6)
        assert np.allclose(orig_pos, re_pos, atol=1e-6)

    art_nz = export_certificate_artifact(st_nz, certs_nz, validator=kernel.validator)
    assert art_nz.schema_version == "1.1.1"
    assert art_nz.catalog_sha256 == PINNED_RELEASE_CATALOG_SHA256_V111
    assert art_nz.ground_plane_z_ldu == -24.0

    # Default VerificationObligations() expects ground_plane_z_ldu == 0.0 -> MUST REJECT nonzero datum!
    rep_default_reject = verify_certificate_independently(art_nz)
    assert rep_default_reject.verified is False
    assert any("ground_plane_z_ldu_mismatch" in r for r in rep_default_reject.failure_reasons)

    # Explicit expected_ground_plane_z_ldu=-24.0 -> PASS!
    rep_nz_ok = verify_certificate_independently(
        art_nz,
        VerificationObligations(expected_ground_plane_z_ldu=-24.0),
    )
    assert rep_nz_ok.verified is True, f"Nonzero datum verification failed: {rep_nz_ok.failure_reasons}"

    nz_json = tmp_path / "cert_z_neg24.json"
    art_nz.save_json(nz_json)
    assert verifier_cli_main([str(nz_json)]) == 1
    assert verifier_cli_main([str(nz_json), "--expected-ground-plane-z-ldu", "-24.0"]) == 0

    # Forged `ground_plane_z_ldu = 100.0` floating brick rejected under default obligations
    st_float = AssemblyState()
    st_float.place_root_part("3001", LDUFrame(0, (0, 0, 100)), is_grounded=True)
    certs_forged = kernel.certify(st_float, foundation_policy="table_friction_only", ground_plane_z_ldu=100.0)
    art_forged = export_certificate_artifact(st_float, certs_forged, validator=kernel.validator)
    rep_forged = verify_certificate_independently(art_forged)
    assert rep_forged.verified is False
    assert any("ground_plane_z_ldu_mismatch" in r for r in rep_forged.failure_reasons)

    # 2. Backward compatibility: legacy Schema "1.1.0" artifact (without `ground_plane_z_ldu` field)
    st_z0 = AssemblyState()
    st_z0.place_root_part("3001", LDUFrame(0, (0, 0, 0)), is_grounded=True)
    st_z0.place_root_part("3001", LDUFrame(0, (0, 0, 24)), is_grounded=False)
    certs_z0 = kernel.certify(st_z0, foundation_policy="table_friction_only")
    art_z0 = export_certificate_artifact(st_z0, certs_z0, validator=kernel.validator)
    legacy_d = art_z0.to_json_dict()
    legacy_d["schema_version"] = "1.1.0"
    legacy_d["catalog_sha256"] = PINNED_RELEASE_CATALOG_SHA256_V110
    legacy_d.pop("ground_plane_z_ldu", None)
    legacy_art_110 = VersionedCertificateArtifact.from_json_dict(legacy_d)
    assert legacy_art_110.ground_plane_z_ldu == 0.0
    rep_110 = verify_certificate_independently(legacy_art_110)
    assert rep_110.verified is True, f"Legacy 1.1.0 replay failed: {rep_110.failure_reasons}"
