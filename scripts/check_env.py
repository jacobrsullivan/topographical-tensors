#!/usr/bin/env python3
"""Verify hardware, PyTorch MPS/CUDA backend, Stockfish binary, and data staging directories."""

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
    print(f"Hostname:       {platform.node()}")
    print(f"Platform:       {platform.platform()} ({platform.machine()})")
    print(f"Python:         {sys.version.split()[0]} ({sys.executable})")

    try:
        import torch

        mps_avail = torch.backends.mps.is_available()
        mps_built = torch.backends.mps.is_built()
        print(f"PyTorch:        {torch.__version__} (MPS built={mps_built}, available={mps_avail})")
    except ImportError:
        print("PyTorch:        NOT INSTALLED")

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
