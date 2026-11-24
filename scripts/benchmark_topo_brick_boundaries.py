#!/usr/bin/env python3
"""
Boundary & Cold-Start Benchmark CLI (`Phase 15 Priority 5`, `v5.3.0`).

Measures in fresh isolated Python subprocesses (pinned to single-core execution):
  1. Cold `import topo_brick` latency (`"torch" not in sys.modules`),
  2. Cold `from topo_brick.api import TopoBrickKernel, ...` resolution across all
     `SUPPORTED_CONSUMER_EXPORTS` (`"torch" not in sys.modules`),
  3. First `CATALOG_RETRIEVAL_INDEX.funnel_metadata()` load (`16,085` catalog IDs -> `1,636`
     active buildable molds),
  4. First deterministic sculpted `TopoBrickKernel().realize(...)` + `"assembly_and_target"`
     independent replay (`"torch" not in sys.modules`),
  5. Isolated `python -m topo_brick.evaluation.independent_verifier --self-test` (`0` kernel
     imports and `"torch" not in sys.modules`),
  6. Pinned asset provenance & Schema `1.1.1` / `1.1.0` / `1.0.0` verification check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _single_core_env() -> Dict[str, str]:
    env = dict(os.environ)
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[k] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing_pp}" if existing_pp else str(SRC_DIR)
    return env


def _run_probe(code: str, timeout_s: float = 60.0) -> Dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=_single_core_env(),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=True,
    )
    proc_ms = (time.perf_counter() - t0) * 1000.0
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["subprocess_total_ms"] = round(proc_ms, 2)
    return payload


def run_boundary_benchmarks() -> Dict[str, Any]:
    # 1. Cold `import topo_brick`
    p1 = _run_probe(
        """
import json, sys, time
t0 = time.perf_counter()
import topo_brick
ms = (time.perf_counter() - t0) * 1000.0
print(json.dumps({
    "in_process_ms": round(ms, 3),
    "version": topo_brick.__version__,
    "torch_loaded": "torch" in sys.modules,
}))
"""
    )

    # 2. Resolve every symbol in `topo_brick.api.SUPPORTED_CONSUMER_EXPORTS`
    p2 = _run_probe(
        """
import json, sys, time
t0 = time.perf_counter()
import topo_brick.api as api
resolved = [getattr(api, name) for name in api.SUPPORTED_CONSUMER_EXPORTS]
ms = (time.perf_counter() - t0) * 1000.0
print(json.dumps({
    "in_process_ms": round(ms, 3),
    "export_count": len(resolved),
    "torch_loaded": "torch" in sys.modules,
}))
"""
    )

    # 3. Cold `CATALOG_RETRIEVAL_INDEX.funnel_metadata()` load
    p3 = _run_probe(
        """
import json, sys, time
t0 = time.perf_counter()
from topo_brick.interface.catalog_retrieval_index import CATALOG_RETRIEVAL_INDEX
meta = CATALOG_RETRIEVAL_INDEX.funnel_metadata()
ms = (time.perf_counter() - t0) * 1000.0
print(json.dumps({
    "in_process_ms": round(ms, 3),
    "funnel": meta,
    "torch_loaded": "torch" in sys.modules,
}))
"""
    )

    # 4. Deterministic sculpted `TopoBrickKernel.realize()` + `"assembly_and_target"` replay without `torch`
    p4 = _run_probe(
        """
import json, sys, time
import numpy as np
t0 = time.perf_counter()
from topo_brick.api import (
    AdaptiveSurfaceTarget,
    TopoBrickKernel,
    build_relying_party_obligations,
    enforce_and_audit_single_core,
    export_certificate_artifact,
    verify_certificate_independently,
    voxelize_triangle_mesh_to_target_field,
)
audit = enforce_and_audit_single_core()
verts = np.array([
    [0.0, 0.0, 0.0], [80.0, 0.0, 0.0], [80.0, 40.0, 0.0], [0.0, 40.0, 0.0],
    [10.0, 0.0, 48.0], [70.0, 0.0, 48.0], [70.0, 40.0, 48.0], [10.0, 40.0, 48.0],
], dtype=np.float64)
faces = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7],
], dtype=np.int32)
ctf, _ = voxelize_triangle_mesh_to_target_field(verts, faces, max_xy_studs=4, max_z_courses=2)
ad = AdaptiveSurfaceTarget.from_triangle_mesh(verts, faces, max_xy_studs=4, max_z_courses=2)
kernel = TopoBrickKernel()
res = kernel.realize(
    ctf,
    part_retrieval="open_catalog_top_k",
    retrieval_top_k=24,
    surface_refinement="open_catalog_sculpted",
    adaptive_mesh_target=ad,
)
eff_tf = res.evaluated_target_field if res.evaluated_target_field is not None else ctf
art = export_certificate_artifact(
    res.assembly,
    res.certificates,
    validator=kernel.validator,
    target_field=eff_tf,
    foundation_policy=res.route_provenance.foundation_policy,
    support_policy="none",
)
ob = build_relying_party_obligations(
    expected_target_field=eff_tf,
    min_target_iou=max(0.10, res.route_provenance.min_recall_threshold * 0.5),
    foundation_policy=res.route_provenance.foundation_policy,
)
ver = verify_certificate_independently(art, ob)
ms = (time.perf_counter() - t0) * 1000.0
print(json.dumps({
    "in_process_ms": round(ms, 3),
    "certified": bool(res.certificates.overall_certified),
    "verified": bool(ver.verified),
    "verification_scope": ver.verification_scope,
    "num_parts": int(res.assembly.num_parts),
    "single_core_compliant": bool(audit.strictly_single_core_verified),
    "torch_loaded": "torch" in sys.modules,
}))
"""
    )

    # 5. Isolated `independent_verifier` self-test & zero-kernel-import audit
    p5 = _run_probe(
        """
import json, sys, time
t0 = time.perf_counter()
from topo_brick.evaluation import independent_verifier as iv
rc = iv.main(["--self-test"])
ms = (time.perf_counter() - t0) * 1000.0
kernel_mods = [m for m in sys.modules if m == "topo_brick.kernel" or m.startswith("topo_brick.kernel.")]
print(json.dumps({
    "in_process_ms": round(ms, 3),
    "exit_code": int(rc),
    "kernel_modules_loaded": kernel_mods,
    "torch_loaded": "torch" in sys.modules,
}))
"""
    )

    # 6. Pinned release catalog provenance & schema compatibility (`1.0.0`, `1.1.0`, `1.1.1`)
    p6 = _run_probe(
        """
import json, time
t0 = time.perf_counter()
from topo_brick.evaluation.independent_verifier import (
    CERTIFICATE_SCHEMA_VERSION,
    SUPPORTED_CERTIFICATE_SCHEMAS,
    compute_catalog_sha256,
)
from topo_brick.interface.release_provenance import (
    PINNED_RELEASE_CATALOG_SHA256_V111,
    verify_bundled_asset_digests,
)
prov = verify_bundled_asset_digests()
cat_sha = compute_catalog_sha256()
ms = (time.perf_counter() - t0) * 1000.0
print(json.dumps({
    "in_process_ms": round(ms, 3),
    "schema_version": CERTIFICATE_SCHEMA_VERSION,
    "supported_schemas": list(SUPPORTED_CERTIFICATE_SCHEMAS),
    "assets_verified_count": len(prov),
    "catalog_sha256_matches_v111": cat_sha == PINNED_RELEASE_CATALOG_SHA256_V111,
}))
"""
    )

    # 7. Hard `sys.meta_path` Torch/Transformers/Chess-blocked lightweight consumer workflow (`Phase 16 §5`)
    p7 = _run_probe(
        """
import contextlib, importlib.abc, io, json, runpy, sys, time
BLOCKED = ("torch", "transformers", "chess")
class _BlockHeavyExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root in BLOCKED:
            raise ModuleNotFoundError(f"Blocked optional extra {fullname!r} in lightweight core test")
        return None
sys.meta_path.insert(0, _BlockHeavyExtras())

t0 = time.perf_counter()
import topo_brick.api as api
resolved = [getattr(api, name) for name in api.SUPPORTED_CONSUMER_EXPORTS]
box_tf = api.TargetShapeBoxes(boxes=(api.LDUBox(-40, 20, 0, 40, 60, 24),))
kern = api.TopoBrickKernel()
res = kern.realize(box_tf, solver_id="S1_CoverAndRepair")
mpd_txt = res.assembly.to_ldraw_mpd()
art = api.export_certificate_artifact(
    res.assembly, res.certificates, validator=kern.validator, target_field=box_tf
)
_, rt_rep = api.verify_exported_mpd_roundtrip(
    mpd_txt, expected_assembly=res.assembly, certificate_artifact=art
)
ver = api.verify_certificate_independently(
    art, api.build_relying_party_obligations(expected_target_field=box_tf, min_target_iou=0.95)
)
with contextlib.redirect_stdout(io.StringIO()):
    qs_ns = runpy.run_path("scripts/run_topo_brick_quickstart.py")
    qs_rc = int(qs_ns["main"]())
ms = (time.perf_counter() - t0) * 1000.0
print(json.dumps({
    "in_process_ms": round(ms, 3),
    "export_count": len(resolved),
    "certified": bool(res.certificates.overall_certified),
    "mpd_roundtrip_verified": bool(rt_rep.roundtrip_verified),
    "independent_verified": bool(ver.verified),
    "quickstart_rc": qs_rc,
    "blocked_loaded": [m for m in BLOCKED if m in sys.modules],
}))
"""
    )

    all_passed = bool(
        not p1["torch_loaded"]
        and not p2["torch_loaded"]
        and not p3["torch_loaded"]
        and not p4["torch_loaded"]
        and p4["certified"]
        and p4["verified"]
        and p4["verification_scope"] == "assembly_and_target"
        and p5["exit_code"] == 0
        and len(p5["kernel_modules_loaded"]) == 0
        and not p5["torch_loaded"]
        and p6["catalog_sha256_matches_v111"]
        and p7["certified"]
        and p7["mpd_roundtrip_verified"]
        and p7["independent_verified"]
        and p7["quickstart_rc"] == 0
        and len(p7["blocked_loaded"]) == 0
    )
    return {
        "all_boundaries_passed": all_passed,
        "cold_import_topo_brick": p1,
        "consumer_api_surface": p2,
        "catalog_funnel_load": p3,
        "torch_free_sculpted_realize_and_verify": p4,
        "isolated_verifier_self_test": p5,
        "release_provenance_and_schemas": p6,
        "torch_blocked_lightweight_core_workflow": p7,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark topo_brick architectural & import boundaries.")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write JSON report.")
    args = parser.parse_args(argv)

    report = run_boundary_benchmarks()
    txt = json.dumps(report, indent=2, sort_keys=True)
    print(txt)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(txt + "\n", encoding="utf-8")
    return 0 if report["all_boundaries_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
