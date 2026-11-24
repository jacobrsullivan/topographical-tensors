"""
Phase 15 Golden Equivalence Suite (`Priority 3`):
Verifies bit-for-bit preservation of placed `(part_id, pose, color_id, is_grounded)`,
`ExecutionTrace` operations, evaluated target SHA-256, IoU/recall, and LP solver/witness
counts across `6` canonical realization routes throughout module & stage extractions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import topo_brick as tb

REPO_ROOT = Path(__file__).resolve().parents[2]

FROZEN_V521_SNAPSHOTS: Dict[str, Dict[str, Any]] = {
    "route_01_course_s1_8part": {
        "status": "COMPLETED_CERTIFIED",
        "num_parts": 2,
        "parts_sha256": "26cd6ca4a7788f3238f8e75599495ec82420d8bee76de9719e7bfd780405446a",
        "ops_sha256": "bc74d4a8967fa6622c93b93bee03a613c5454b44fec7f2d325a820772b318dcf",
        "target_sha256": "aea9f15032cb9b5757d538ce2dd7602c3f0692ecbb7e1fe3426d676994b7aede",
        "target_iou": 1.0,
        "target_recall": 1.0,
        "lp_calls": 1,
        "witness_hits": 0,
    },
    "route_02_macro_block_biplane": {
        "status": "COMPLETED_CERTIFIED",
        "num_parts": 41,
        "parts_sha256": "85da368388782f950096ffc69165c038a7b3efb640f725f4b7d84055d2891320",
        "ops_sha256": "23874e19adb12aa571c02741357f542f555c262e7bdab3e38200ddeefde79068",
        "target_sha256": "344c86bb61ca5e1760e337688b1c87917cb5d9850f6909e42d8e17c72d794250",
        "target_iou": 0.935897,
        "target_recall": 0.960526,
        "lp_calls": 7,
        "witness_hits": 72,
    },
    "route_03_portal_fast_path_sculpted": {
        "status": "COMPLETED_CERTIFIED",
        "num_parts": 16,
        "parts_sha256": "0185cfdb8e4bd5b06a14d6bb5d1431299950a479815591b76b4f041ddeb575af",
        "ops_sha256": "be96a9af91526bcbab311198aaa70067b86ed50379b14e689beee1b058920a85",
        "target_sha256": "4fb121e6bcb1e09e72a37ae1530d02c704cfad3ecfd5a595594791e5b1583ca9",
        "target_iou": 0.965385,
        "target_recall": 0.984314,
        "lp_calls": 2,
        "witness_hits": 27,
    },
    "route_04_sculpted_colored_set_10698": {
        "status": "COMPLETED_CERTIFIED",
        "num_parts": 18,
        "parts_sha256": "6f919186c97450447e71309ecc9ccbf784e03c97e60fec4525732793e4df8c61",
        "ops_sha256": "2037e02253efc8608f46824535dcd7d8c762a068f87f5125101b7c9404579db2",
        "target_sha256": "2cba6c437dd868ae10cd1d4eea22609fad3b853f17d27aa8e2c2a99bcad881b7",
        "target_iou": 0.648179,
        "target_recall": 0.756118,
        "lp_calls": 28,
        "witness_hits": 18,
    },
    "route_05_hollow_shell_tower": {
        "status": "COMPLETED_CERTIFIED",
        "num_parts": 30,
        "parts_sha256": "6e380160230810dc2af1bfe6972b53240c3154623b94b4072264d8743be04a74",
        "ops_sha256": "e82702359cc22ce3ae6cc73fb450703dd7eeac3d27cff2dabe3464b6caa590fa",
        "target_sha256": "fb47e0caf8eb268e6c52034521591c657868c12c7a2a831fee0c6adca9ae088e",
        "target_iou": 1.0,
        "target_recall": 1.0,
        "lp_calls": 1,
        "witness_hits": 51,
    },
    "route_06_compositional_mpd_3node": {
        "status": "COMPLETED_CERTIFIED",
        "num_parts": 18,
        "parts_sha256": "950d87987ac08764faed91d5626d18904ea12a80fc7e92f5def59a3c51acf573",
        "ops_sha256": "55414923ec4e0dfc3d5eef489e9948d1690967e5cd2c609868590d1659dfced7",
        "target_sha256": "c890dfb40b615e846821db1ea9737366958aad2b6610d52098670c5cf482fef2",
        "target_iou": 0.991026,
        "target_recall": 1.0,
        "lp_calls": 11,
        "witness_hits": 22,
    },
}


def _box_mesh(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray]:
    w, d, h = float(nx * 20), float(ny * 20), float(nz * 24)
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [w, 0.0, 0.0],
            [w, d, 0.0],
            [0.0, d, 0.0],
            [0.0, 0.0, h],
            [w, 0.0, h],
            [w, d, h],
            [0.0, d, h],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return verts, faces


def _snapshot_result(kernel: tb.TopoBrickKernel, res: tb.KernelRealizeResult) -> Dict[str, Any]:
    parts = [
        {
            "iid": int(iid),
            "pid": str(p.part_id),
            "rot": int(p.pose.rot_id),
            "pos": [int(c) for c in p.pose.pos],
            "col": int(p.color_id),
            "gr": bool(p.is_grounded),
        }
        for iid, p in sorted(res.assembly.placed_parts.items())
    ]
    parts_sha = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    art = tb.export_certificate_artifact(
        res.assembly,
        res.certificates,
        validator=kernel.validator,
        target_field=res.evaluated_target_field,
    )
    ops_sha = hashlib.sha256(
        json.dumps(art.execution_operations, sort_keys=True).encode()
    ).hexdigest()
    return {
        "status": res.status.value,
        "num_parts": int(res.assembly.num_parts),
        "parts_sha256": parts_sha,
        "ops_sha256": ops_sha,
        "target_sha256": art.target_sha256,
        "target_iou": round(float(res.build_record.target_iou), 6),
        "target_recall": round(float(res.build_record.target_recall), 6),
        "lp_calls": int(res.build_record.lp_solver_calls),
        "witness_hits": int(res.build_record.witness_reuse_hits),
    }


def test_p15_six_route_golden_equivalence_snapshots() -> None:
    """Verify zero drift across all 6 canonical realization routes after structural refactoring."""
    kernel = tb.TopoBrickKernel()
    corpus, _ = tb.load_frozen_80_unseen_mesh_corpus()

    # 1. S1 course search
    tf1 = tb.TargetShapeBoxes(
        boxes=(
            tb.LDUBox(0, 0, 0, 80, 40, 24),
            tb.LDUBox(0, 0, 24, 80, 40, 48),
        )
    )
    s1 = _snapshot_result(
        kernel,
        kernel.realize(tf1, solver_id="S1_CoverAndRepair", support_policy="minimal_permanent_pillars"),
    )
    assert s1 == FROZEN_V521_SNAPSHOTS["route_01_course_s1_8part"]

    # 2. Macro-block cantilevered tabletop biplane
    biplane_boxes = (
        tb.LDUBox(120, 40, 0, 200, 160, 24),
        tb.LDUBox(100, 0, 24, 220, 200, 72),
        tb.LDUBox(100, 60, 72, 220, 140, 96),
        tb.LDUBox(0, 60, 24, 320, 140, 48),
        tb.LDUBox(20, 60, 48, 60, 140, 96),
        tb.LDUBox(260, 60, 48, 300, 140, 96),
        tb.LDUBox(0, 60, 96, 320, 140, 120),
    )
    interplane_gap_boxes = (
        tb.LDUBox(60, 60, 48, 100, 140, 96),
        tb.LDUBox(220, 60, 48, 260, 140, 96),
    )
    s2 = _snapshot_result(
        kernel,
        kernel.realize(
            target_field=tb.SparseTiledTargetField(boxes=biplane_boxes),
            macro_block_mode="force",
            foundation_policy="table_friction_only",
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=interplane_gap_boxes,
        ),
    )
    assert s2 == FROZEN_V521_SNAPSHOTS["route_02_macro_block_biplane"]

    # 3. Portal fast-path sculpted mesh (quickstart fixture)
    spec3 = next(m for m in corpus if len(m["required_openings_ldu"]) > 0)
    ctf3, _ = tb.voxelize_triangle_mesh_to_target_field(
        spec3["verts"],
        spec3["faces"],
        max_xy_studs=int(spec3["max_xy_studs"]),
        max_z_courses=int(spec3["max_z_courses"]),
    )
    ad3 = tb.AdaptiveSurfaceTarget.from_triangle_mesh(
        spec3["verts"],
        spec3["faces"],
        max_xy_studs=int(spec3["max_xy_studs"]),
        max_z_courses=int(spec3["max_z_courses"]),
    )
    v3 = ad3.mesh_surf_pts
    zn3 = (v3[:, 2] - float(v3[:, 2].min())) / max(1.0, float(np.ptp(v3[:, 2])))
    rgb3 = np.where(
        zn3[:, None] < 0.5,
        np.array([[160, 165, 169]], dtype=np.uint8),
        np.array([[180, 0, 0]], dtype=np.uint8),
    )
    s3 = _snapshot_result(
        kernel,
        kernel.realize(
            ctf3,
            part_retrieval="open_catalog_top_k",
            retrieval_top_k=32,
            license_filter="cc_by_4_0_only",
            semantic_style_hint="smooth_studless",
            surface_refinement="open_catalog_sculpted",
            adaptive_mesh_target=ad3,
            support_policy="minimal_permanent_pillars",
            forbidden_support_boxes=tuple(spec3["required_openings_ldu"]),
            vertex_colors_rgb=rgb3,
            verts_ldu=v3,
        ),
    )
    assert s3 == FROZEN_V521_SNAPSHOTS["route_03_portal_fast_path_sculpted"]

    # 4. Sculpted colored inventory (Set 10698-1) + ViewPrioritySpec
    spec4 = next(
        m for m in corpus if len(m["required_openings_ldu"]) == 0 and int(m["max_xy_studs"]) <= 8
    )
    ctf4, _ = tb.voxelize_triangle_mesh_to_target_field(
        spec4["verts"],
        spec4["faces"],
        max_xy_studs=int(spec4["max_xy_studs"]),
        max_z_courses=int(spec4["max_z_courses"]),
    )
    ad4 = tb.AdaptiveSurfaceTarget.from_triangle_mesh(
        spec4["verts"],
        spec4["faces"],
        max_xy_studs=int(spec4["max_xy_studs"]),
        max_z_courses=int(spec4["max_z_courses"]),
    )
    v4 = ad4.mesh_surf_pts
    rgb4 = np.tile(np.array([[220, 180, 20]], dtype=np.uint8), (len(v4), 1))
    s4 = _snapshot_result(
        kernel,
        kernel.realize(
            ctf4,
            part_retrieval="open_catalog_top_k",
            surface_refinement="open_catalog_sculpted",
            adaptive_mesh_target=ad4,
            colored_inventory=tb.SET_10698_COLORED_INVENTORY,
            view_priority=tb.ViewPrioritySpec(view_dir_ldu=(1.0, 1.0, -0.5)),
            vertex_colors_rgb=rgb4,
            verts_ldu=v4,
        ),
    )
    assert s4 == FROZEN_V521_SNAPSHOTS["route_04_sculpted_colored_set_10698"]

    # 5. Hollow shell tower
    vb5, fb5 = _box_mesh(10, 10, 6)
    ctf5, _ = tb.voxelize_triangle_mesh_to_target_field(vb5, fb5, max_xy_studs=10, max_z_courses=6)
    ad5 = tb.AdaptiveSurfaceTarget.from_triangle_mesh(vb5, fb5, max_xy_studs=10, max_z_courses=6)
    s5 = _snapshot_result(
        kernel,
        kernel.realize(
            ctf5,
            surface_refinement="open_catalog_sculpted",
            adaptive_mesh_target=ad5,
            hollow_shell_policy=tb.HollowShellPolicy(
                enabled=True, wall_thickness_studs=2, roof_thickness_plates=3
            ),
        ),
    )
    assert s5 == FROZEN_V521_SNAPSHOTS["route_05_hollow_shell_tower"]

    # 6. Compositional 3-node MPD + landmark
    vb_base, fb_base = _box_mesh(6, 4, 2)
    ad_base = tb.AdaptiveSurfaceTarget.from_triangle_mesh(
        vb_base, fb_base, max_xy_studs=6, max_z_courses=2
    )
    vb_tur, fb_tur = _box_mesh(4, 2, 1)
    ad_tur = tb.AdaptiveSurfaceTarget.from_triangle_mesh(
        vb_tur, fb_tur, max_xy_studs=4, max_z_courses=1
    )
    vb_cap, fb_cap = _box_mesh(2, 2, 1)
    ad_cap = tb.AdaptiveSurfaceTarget.from_triangle_mesh(
        vb_cap, fb_cap, max_xy_studs=2, max_z_courses=1
    )
    ctf6, _ = tb.voxelize_triangle_mesh_to_target_field(
        vb_base, fb_base, max_xy_studs=6, max_z_courses=2
    )
    graph6 = (
        tb.SubmodelNodeSpec(
            submodel_name="hull.ldr", adaptive_target=ad_base, world_offset_ldu=(0, 0, 0)
        ),
        tb.SubmodelNodeSpec(
            submodel_name="turret.ldr",
            adaptive_target=ad_tur,
            world_offset_ldu=(20, 20, 48),
            parent_submodel="hull.ldr",
        ),
        tb.SubmodelNodeSpec(
            submodel_name="cap.ldr",
            adaptive_target=ad_cap,
            world_offset_ldu=(40, 20, 72),
            parent_submodel="turret.ldr",
        ),
    )
    lm6 = (
        tb.LandmarkAttachmentRequest(
            role="BAR_POLE_ANTENNA",
            anchor_box_ldu=tb.LDUBox(40, 20, 96, 80, 60, 120),
            preferred_color_id=4,
        ),
    )
    s6 = _snapshot_result(
        kernel,
        kernel.realize(ctf6, subassembly_graph=graph6, landmark_requests=lm6),
    )
    assert s6 == FROZEN_V521_SNAPSHOTS["route_06_compositional_mpd_3node"]
