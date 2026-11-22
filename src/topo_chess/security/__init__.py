"""Security primitives for authenticated, restricted checkpoint I/O (`topo_chess`)."""

from topo_chess.security.checkpoint_io import (
    MIN_SAFE_TORCH_VERSION,
    CheckpointIntegrityError,
    CheckpointSchemaError,
    CheckpointSecurityError,
    UnsupportedTorchSecurityVersionError,
    create_private_checkpoint_dir,
    parse_torch_version_tuple,
    safe_load_authenticated_checkpoint,
    safe_save_checkpoint,
    validate_phase_c_checkpoint_schema,
    validate_state_dict_tensors,
    validate_wp4_state_bundle,
    verify_safe_torch_version,
)

__all__ = [
    "MIN_SAFE_TORCH_VERSION",
    "CheckpointIntegrityError",
    "CheckpointSchemaError",
    "CheckpointSecurityError",
    "UnsupportedTorchSecurityVersionError",
    "create_private_checkpoint_dir",
    "parse_torch_version_tuple",
    "safe_load_authenticated_checkpoint",
    "safe_save_checkpoint",
    "validate_phase_c_checkpoint_schema",
    "validate_state_dict_tensors",
    "validate_wp4_state_bundle",
    "verify_safe_torch_version",
]
