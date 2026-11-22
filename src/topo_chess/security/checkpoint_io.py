"""
Authenticated, restricted-deserialization checkpoint I/O (`Phase 13` Security Remediation `S1`, `S2`, `S7`).

Security guarantees:
1. Enforces `torch >= 2.10.0` at runtime (`CVE-2025-32434` and `CVE-2026-24747` remediated).
2. Writes checkpoints into private `0o700` per-run directories using `0o600` atomic temp-file replacement,
   normalizing scalar metadata to built-in Python primitives so `weights_only=True` requires zero custom globals.
3. Reads checkpoint bytes once via `O_RDONLY | O_NOFOLLOW`, verifies regular-file byte bounds and optional
   SHA-256 digest via `hmac.compare_digest` (explicit `if ... raise`, never `assert`, active under `python -O`)
   BEFORE invoking `torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)`.
4. Validates model dimension bounds and state-dict tensor shapes/dtypes/finiteness on CPU BEFORE allocating
   `nn.Module` instances.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import hmac
import io
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple
import uuid

import numpy as np
import torch

MIN_SAFE_TORCH_VERSION: Tuple[int, int, int] = (2, 10, 0)
DEFAULT_MAX_CHECKPOINT_BYTES: int = 512 * 1024 * 1024  # 512 MiB
VALID_ABLATION_CONFIGS = frozenset({"Config_A", "Config_B", "Config_C", "Config_D"})
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


class CheckpointSecurityError(RuntimeError):
    """Base error raised on any checkpoint security, provenance, or resource-bound failure."""


class UnsupportedTorchSecurityVersionError(CheckpointSecurityError):
    """Raised when the installed PyTorch version predates restricted-loader security fixes."""


class CheckpointIntegrityError(CheckpointSecurityError):
    """Raised when a checkpoint fails SHA-256 authentication or file-type/size checks."""


class CheckpointSchemaError(CheckpointSecurityError):
    """Raised when a deserialized checkpoint violates hyperparameter or tensor bounds."""


def parse_torch_version_tuple(version_str: Optional[str] = None) -> Tuple[int, int, int]:
    """Parse `'major.minor.patch'` from `torch.__version__` (ignoring `+cpu` / `rc` suffixes)."""
    raw = str(torch.__version__ if version_str is None else version_str).strip()
    m = _VERSION_RE.match(raw)
    if m is None:
        raise UnsupportedTorchSecurityVersionError(
            f"Unable to parse PyTorch version string: {raw!r}"
        )
    major = int(m.group(1))
    minor = int(m.group(2))
    patch = int(m.group(3)) if m.group(3) is not None else 0
    return (major, minor, patch)


def verify_safe_torch_version(
    version_str: Optional[str] = None,
    min_version: Tuple[int, int, int] = MIN_SAFE_TORCH_VERSION,
) -> Tuple[int, int, int]:
    """Ensure PyTorch meets the minimum reviewed version (`>= 2.10.0`)."""
    parsed = parse_torch_version_tuple(version_str)
    if parsed < min_version:
        req = ".".join(str(x) for x in min_version)
        got = ".".join(str(x) for x in parsed)
        raise UnsupportedTorchSecurityVersionError(
            f"PyTorch {got} is below minimum required security floor {req} "
            "(CVE-2025-32434 / CVE-2026-24747)."
        )
    return parsed


def create_private_checkpoint_dir(base_dir: Path, run_id: Optional[str] = None) -> Path:
    """
    Create and validate a private (`0o700`) per-run checkpoint directory inside `base_dir`
    so concurrent principals sharing a parent path cannot swap checkpoint files.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    eff_id = run_id if run_id else f"run_{os.getpid()}_{uuid.uuid4().hex[:12]}"
    if "/" in eff_id or "\\" in eff_id or ".." in eff_id:
        raise CheckpointSecurityError(f"Invalid checkpoint run_id: {eff_id!r}")
    run_dir = base / eff_id
    run_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
    os.chmod(run_dir, 0o700)
    st = os.lstat(run_dir)
    if not stat.S_ISDIR(st.st_mode) or stat.S_ISLNK(st.st_mode):
        raise CheckpointSecurityError(f"Checkpoint run path is not a real directory: {run_dir}")
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        raise CheckpointSecurityError(
            f"Checkpoint directory {run_dir} is owned by uid={st.st_uid}, expected {os.getuid()}"
        )
    if (st.st_mode & 0o077) != 0:
        raise CheckpointSecurityError(
            f"Checkpoint directory {run_dir} has overly permissive mode {oct(st.st_mode & 0o777)}"
        )
    return run_dir


def _sanitize_checkpoint_tree(obj: Any) -> Any:
    """Convert NumPy scalars / OrderedDicts into restricted-loader-safe Python & Tensor primitives."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise CheckpointSchemaError(f"Non-finite scalar float in checkpoint payload: {obj}")
        return float(obj)
    if isinstance(obj, np.generic):
        return _sanitize_checkpoint_tree(obj.item())
    if isinstance(obj, torch.Tensor):
        t_cpu = obj.detach().cpu()
        if (t_cpu.is_floating_point() or t_cpu.is_complex()) and not bool(torch.isfinite(t_cpu).all()):
            raise CheckpointSchemaError("Non-finite tensor values in checkpoint payload.")
        return t_cpu
    if isinstance(obj, OrderedDict):
        return OrderedDict((str(k), _sanitize_checkpoint_tree(v)) for k, v in obj.items())
    if isinstance(obj, Mapping):
        return {str(k): _sanitize_checkpoint_tree(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_sanitize_checkpoint_tree(x) for x in obj)
    if isinstance(obj, list):
        return [_sanitize_checkpoint_tree(x) for x in obj]
    raise CheckpointSchemaError(f"Unsupported type in restricted checkpoint payload: {type(obj).__name__}")


def safe_save_checkpoint(payload: Mapping[str, Any], dest_path: str | Path) -> Tuple[str, int]:
    """
    Atomically serialize a sanitized checkpoint to `dest_path` (`0o600`) and return
    `(sha256_hex, size_bytes)` computed over the exact serialized byte buffer.
    """
    verify_safe_torch_version()
    clean_payload = _sanitize_checkpoint_tree(payload)
    buf = io.BytesIO()
    torch.save(clean_payload, buf)
    raw_bytes = buf.getvalue()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".ckpt_", suffix=".tmp", dir=str(dest.parent))
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(raw_bytes)
        written = 0
        while written < len(raw_bytes):
            n_w = os.write(fd, view[written:])
            if n_w <= 0:
                raise OSError("Short write while saving checkpoint")
            written += n_w
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(tmp_path, dest)
    except Exception:
        if fd >= 0:
            os.close(fd)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return sha256_hex, len(raw_bytes)


def _read_regular_file_once(path: Path, max_bytes: int) -> bytes:
    """Open `path` without following symlinks, verify regular-file bounds, and read bytes once."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise CheckpointIntegrityError(f"Cannot safely open checkpoint {path}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise CheckpointIntegrityError(f"Checkpoint path is not a regular file: {path}")
        if st.st_size <= 0:
            raise CheckpointIntegrityError(f"Checkpoint file is empty: {path}")
        if st.st_size > max_bytes:
            raise CheckpointIntegrityError(
                f"Checkpoint size {st.st_size} exceeds limit {max_bytes} bytes: {path}"
            )
        chunks = []
        total = 0
        while True:
            blk = os.read(fd, 1024 * 1024)
            if not blk:
                break
            total += len(blk)
            if total > max_bytes:
                raise CheckpointIntegrityError(
                    f"Checkpoint grew beyond {max_bytes} bytes during read: {path}"
                )
            chunks.append(blk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def safe_load_authenticated_checkpoint(
    path: str | Path,
    *,
    expected_sha256: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
    map_location: str | torch.device = "cpu",
) -> Tuple[Any, str]:
    """
    Read `path` once, authenticate `expected_sha256` BEFORE deserialization (never using `assert`),
    and load via `torch.load(io.BytesIO(raw_bytes), map_location=..., weights_only=True)`.
    Returns `(deserialized_object, actual_sha256_hex)`.
    """
    verify_safe_torch_version()
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    ckpt_path = Path(path)
    raw_bytes = _read_regular_file_once(ckpt_path, max_bytes=max_bytes)
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()

    if expected_sha256 is not None:
        exp_norm = str(expected_sha256).strip().lower()
        if _HEX64_RE.match(exp_norm) is None:
            raise CheckpointIntegrityError(
                f"Malformed expected SHA-256 digest: {expected_sha256!r}"
            )
        if not hmac.compare_digest(actual_sha, exp_norm):
            raise CheckpointIntegrityError(
                f"Checkpoint SHA-256 mismatch for {ckpt_path}: "
                f"expected {exp_norm}, got {actual_sha}"
            )

    try:
        loaded = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointSecurityError(
            f"Restricted weights_only=True deserialization rejected {ckpt_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    target_dev = torch.device(map_location)
    if target_dev.type != "cpu":
        loaded = _move_tensors_to_device(loaded, target_dev)
    return loaded, actual_sha


def _move_tensors_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, _move_tensors_to_device(v, device)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in obj.items()}
    return obj


def validate_state_dict_tensors(
    state_dict: Any,
    *,
    label: str = "state_dict",
    max_tensors: int = 2_048,
    max_dim_size: int = 65_536,
    max_total_elements: int = 50_000_000,
) -> Dict[str, torch.Tensor]:
    """Validate state-dict keys, tensor ranks, element counts, and finiteness before `load_state_dict`."""
    if not isinstance(state_dict, Mapping) or len(state_dict) == 0:
        raise CheckpointSchemaError(f"{label} must be a non-empty mapping of tensors")
    if len(state_dict) > max_tensors:
        raise CheckpointSchemaError(
            f"{label} tensor count {len(state_dict)} exceeds limit {max_tensors}"
        )
    out: Dict[str, torch.Tensor] = {}
    total_elems = 0
    for k, v in state_dict.items():
        if not isinstance(k, str) or not k or len(k) > 256:
            raise CheckpointSchemaError(f"Invalid parameter key in {label}: {k!r}")
        if not isinstance(v, torch.Tensor):
            raise CheckpointSchemaError(
                f"Non-tensor entry {k!r} ({type(v).__name__}) in {label}"
            )
        if v.ndim > 6 or any(int(d) < 0 or int(d) > max_dim_size for d in v.shape):
            raise CheckpointSchemaError(f"Out-of-bounds tensor shape {tuple(v.shape)} for {k!r}")
        n_el = int(v.numel())
        total_elems += n_el
        if total_elems > max_total_elements:
            raise CheckpointSchemaError(
                f"{label} total elements {total_elems} exceed limit {max_total_elements}"
            )
        if (v.is_floating_point() or v.is_complex()) and not bool(torch.isfinite(v).all()):
            raise CheckpointSchemaError(f"Non-finite weights in {label}[{k!r}]")
        out[k] = v
    return out


def validate_phase_c_checkpoint_schema(
    loaded: Any,
    *,
    max_d_model: int = 1024,
    max_layers: int = 32,
    max_heads: int = 32,
) -> Dict[str, Any]:
    """
    Validate Phase C checkpoint metadata and tensor shapes BEFORE allocating `nn.Module`s.
    """
    if not isinstance(loaded, Mapping):
        raise CheckpointSchemaError(f"Expected Phase C checkpoint dict, got {type(loaded).__name__}")
    required_keys = {"config_name", "seed", "d_model", "num_layers", "n_heads", "encoder_state_dict"}
    missing = required_keys - set(loaded.keys())
    if missing:
        raise CheckpointSchemaError(f"Missing Phase C checkpoint keys: {sorted(missing)}")

    cfg_name = loaded["config_name"]
    if not isinstance(cfg_name, str) or cfg_name not in VALID_ABLATION_CONFIGS:
        raise CheckpointSchemaError(f"Invalid Phase C config_name: {cfg_name!r}")

    def _bounded_int(field: str, lo: int, hi: int) -> int:
        val = loaded[field]
        if isinstance(val, bool) or not isinstance(val, int):
            raise CheckpointSchemaError(f"Field {field!r} must be int, got {type(val).__name__}")
        if not (lo <= val <= hi):
            raise CheckpointSchemaError(f"Field {field!r}={val} outside allowed [{lo}, {hi}]")
        return int(val)

    seed = _bounded_int("seed", 0, 2**31 - 1)
    d_model = _bounded_int("d_model", 16, max_d_model)
    num_layers = _bounded_int("num_layers", 1, max_layers)
    n_heads = _bounded_int("n_heads", 1, max_heads)
    if d_model % n_heads != 0:
        raise CheckpointSchemaError(f"d_model={d_model} is not divisible by n_heads={n_heads}")

    enc_sd = validate_state_dict_tensors(loaded["encoder_state_dict"], label="encoder_state_dict")
    raw_pl = loaded.get("planner_state_dict")
    if cfg_name == "Config_D":
        if raw_pl is None:
            raise CheckpointSchemaError("Config_D checkpoint missing required planner_state_dict")
        pl_sd: Optional[Dict[str, torch.Tensor]] = validate_state_dict_tensors(
            raw_pl, label="planner_state_dict"
        )
    else:
        if raw_pl is not None:
            pl_sd = validate_state_dict_tensors(raw_pl, label="planner_state_dict")
        else:
            pl_sd = None

    solve_acc_raw = loaded.get("external_10k_solve_accuracy", 0.0)
    if isinstance(solve_acc_raw, bool) or not isinstance(solve_acc_raw, (int, float)):
        raise CheckpointSchemaError("external_10k_solve_accuracy must be numeric")
    solve_acc = float(solve_acc_raw)
    if not math.isfinite(solve_acc) or not (0.0 <= solve_acc <= 1.0):
        raise CheckpointSchemaError(f"external_10k_solve_accuracy={solve_acc} outside [0, 1]")

    return {
        "config_name": cfg_name,
        "seed": seed,
        "d_model": d_model,
        "num_layers": num_layers,
        "n_heads": n_heads,
        "encoder_state_dict": enc_sd,
        "planner_state_dict": pl_sd,
        "external_10k_solve_accuracy": solve_acc,
    }


def validate_wp4_state_bundle(loaded: Any, *, require_planner: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:
    """Validate WP4/Phase B `{ 'encoder': state_dict, 'd3pm_planner'?: state_dict }` checkpoint bundles."""
    if not isinstance(loaded, Mapping) or "encoder" not in loaded:
        raise CheckpointSchemaError("WP4 checkpoint bundle must contain 'encoder' state_dict")
    allowed = {"encoder", "d3pm_planner"}
    extra = set(loaded.keys()) - allowed
    if extra:
        raise CheckpointSchemaError(f"Unexpected keys in WP4 checkpoint bundle: {sorted(extra)}")
    out: Dict[str, Dict[str, torch.Tensor]] = {
        "encoder": validate_state_dict_tensors(loaded["encoder"], label="encoder")
    }
    if loaded.get("d3pm_planner") is not None:
        out["d3pm_planner"] = validate_state_dict_tensors(loaded["d3pm_planner"], label="d3pm_planner")
    elif require_planner:
        raise CheckpointSchemaError("Config_D WP4 bundle missing 'd3pm_planner' state_dict")
    return out
