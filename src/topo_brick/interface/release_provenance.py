"""
Pinned release provenance & fail-closed catalog integrity primitives (`Phase 13` Security Finding `S8`).

Defends against:
1. Missing `compiled_catalog_manifest.json` or omitted `catalog_gz_sha256` / `coacd_npz_sha256` fields.
2. Joint replacement of `(compiled_ldraw_catalog.json.gz + compiled_catalog_manifest.json)` in a writable
   site-packages installation by anchoring verification to immutable code-level release digests unless an
   operator explicitly supplies `trusted_custom_catalog_sha256`.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class CatalogProvenanceError(ValueError):
    """Raised when compiled catalog metadata or bundled archive digests fail integrity checks."""


PINNED_RELEASE_SET_10698_V110_SHA256: str = (
    "6d9c420c071d027487e447d09e9063649f19266cf02722e39eba878653479152"
)

PINNED_RELEASE_ASSET_DIGESTS_V5_2: Mapping[str, str] = MappingProxyType(
    {
        "compiled_ldraw_catalog.json.gz": "5aae1ca67a4244af5e7c1ef5d87807b39ffbffda58017d039d7ba27154de7fe4",
        "compiled_coacd_hulls.npz": "462de1caebcbe0c9da830174cc887af24906e56de5e0a67b8e275e11b80eba46",
        "compiled_mold_retrieval_index.npz": "40a5773b4eda844bcb875b1737ace1fd222da0fb7f2b7d6c356f711abdee252d",
        "compiled_landmark_detail_index.json.gz": "528a2c1dfa51a4cd74627160d80b8e5823c8433a8c5d5f0995807cb52d941316",
        "set_10698_1_rebrickable_inventory.json": "5990433779ab8e0a8c68d082422e24cefb1183c237f47c8d241a60b6a7a0285e",
    }
)

PINNED_RELEASE_CATALOG_SHA256_V100: str = (
    "a0661ea6c105e66fce921131a77c533106f0193d703977320d778b3929743363"
)
PINNED_RELEASE_CATALOG_SHA256_V110: str = (
    "0e78d354627e90f6af5c1b0c827eae80ed514b7f744458878d3af9d99b0a8681"
)
PINNED_RELEASE_CATALOG_SHA256_V111: str = (
    "e94bda3e7d96448ec319c571629d86a97d23d5c404f3780138327aff40d03e7d"
)

PINNED_RELEASE_CATALOG_SHA256_BY_SCHEMA: Mapping[str, str] = MappingProxyType(
    {
        "1.0.0": PINNED_RELEASE_CATALOG_SHA256_V100,
        "1.1.0": PINNED_RELEASE_CATALOG_SHA256_V110,
        "1.1.1": PINNED_RELEASE_CATALOG_SHA256_V111,
    }
)


def is_valid_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and _HEX64_RE.match(value.strip().lower()) is not None


def validate_compiled_manifest_dict(
    manifest: Any,
    *,
    require_coacd_sha256: bool = True,
    require_structural_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Validate that `compiled_catalog_manifest.json` is present and contains all mandatory
    integrity fields (`catalog_gz_sha256`, `coacd_npz_sha256`, and when `require_structural_metadata=True`,
    `schema_version` and `total_compiled_parts`).
    """
    if not isinstance(manifest, Mapping) or len(manifest) == 0:
        raise CatalogProvenanceError(
            "Missing or empty compiled_catalog_manifest.json; refusing unverified compiled geometry."
        )
    cat_sha = manifest.get("catalog_gz_sha256")
    if not is_valid_sha256_hex(cat_sha):
        raise CatalogProvenanceError(
            f"Manifest missing valid 64-hex 'catalog_gz_sha256': {cat_sha!r}"
        )
    if require_coacd_sha256:
        coacd_sha = manifest.get("coacd_npz_sha256")
        if not is_valid_sha256_hex(coacd_sha):
            raise CatalogProvenanceError(
                f"Manifest missing valid 64-hex 'coacd_npz_sha256': {coacd_sha!r}"
            )
    if require_structural_metadata:
        schema_v = manifest.get("schema_version")
        if not isinstance(schema_v, str) or not schema_v.strip():
            raise CatalogProvenanceError("Manifest missing non-empty 'schema_version'")
        tot_parts = manifest.get("total_compiled_parts")
        if isinstance(tot_parts, bool) or not isinstance(tot_parts, int) or tot_parts <= 0:
            raise CatalogProvenanceError(
                f"Manifest missing positive integer 'total_compiled_parts': {tot_parts!r}"
            )
    return dict(manifest)


def verify_bundled_asset_digests(
    interface_dir: Optional[Path] = None,
    *,
    expected_digests: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Recompute raw SHA-256 digests directly from disk (no stat-cache shortcut) and verify
    every bundled archive against `expected_digests` (defaulting to `PINNED_RELEASE_ASSET_DIGESTS_V5_2`).
    """
    idir = Path(__file__).resolve().parent if interface_dir is None else Path(interface_dir)
    ref_map = PINNED_RELEASE_ASSET_DIGESTS_V5_2 if expected_digests is None else expected_digests
    computed: Dict[str, str] = {}
    for fname, exp_hex in ref_map.items():
        if not is_valid_sha256_hex(exp_hex):
            raise CatalogProvenanceError(f"Invalid expected SHA-256 for {fname}: {exp_hex!r}")
        fpath = idir / fname
        if not fpath.is_file():
            raise CatalogProvenanceError(f"Missing required bundled catalog asset: {fpath}")
        actual_hex = hashlib.sha256(fpath.read_bytes()).hexdigest()
        if not hmac.compare_digest(actual_hex, exp_hex.strip().lower()):
            raise CatalogProvenanceError(
                f"Bundled asset SHA-256 mismatch on {fname}: expected {exp_hex}, got {actual_hex}"
            )
        computed[fname] = actual_hex
    return computed
