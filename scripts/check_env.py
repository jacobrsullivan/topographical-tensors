#!/usr/bin/env python3
"""Verify hardware, PyTorch backend, Stockfish UCI engine, HiGHS/Transformers, and topo_brick catalog."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    print("=" * 72)
    print("Topographical Tensors — Host & Environment Diagnostics")
    print("=" * 72)
    print(f"Platform:       {platform.platform()} ({platform.machine()})")
    print(f"Python:         {sys.version.split()[0]}")

    try:
        import torch

        mps_avail = torch.backends.mps.is_available()
        cuda_avail = torch.cuda.is_available()
        print(f"PyTorch:        {torch.__version__} (CUDA={cuda_avail}, MPS={mps_avail})")
    except ImportError:
        print("PyTorch:        NOT INSTALLED")

    for pkg_name in ("highspy", "transformers", "scipy", "chess"):
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"{pkg_name + ':':16s}{ver}")
        except ImportError:
            print(f"{pkg_name + ':':16s}NOT INSTALLED")

    try:
        from topo_brick import COMBINED_BRICK_CATALOG, compute_catalog_sha256

        n_compiled = len(COMBINED_BRICK_CATALOG)
        cat_sha = compute_catalog_sha256()[:16]
        print(f"topo_brick:     OK ({n_compiled:,} unified parts, catalog_sha256={cat_sha}...)")
    except Exception as exc:
        print(f"topo_brick:     ERROR ({exc})")

    sf_path = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish"
    if Path(sf_path).is_file():
        res = subprocess.run([sf_path, "uci"], input="quit\n", text=True, capture_output=True, timeout=5)
        first_line = res.stdout.splitlines()[0] if res.stdout else "unknown"
        print(f"Stockfish:      {sf_path} ({first_line})")
    else:
        print(f"Stockfish:      NOT FOUND at {sf_path}")

    data_dir = Path(os.environ.get("TOPO_DATA_DIR", "./data"))
    print(f"Data Root:      {data_dir} (exists={data_dir.is_dir()})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
