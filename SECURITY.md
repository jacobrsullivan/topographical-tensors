# Security Policy (`topographical-tensors`)

## Supported Versions

| Version | Supported | Notes |
| :--- | :--- | :--- |
| `>= 5.2.1` | Yes | Enforces `torch>=2.10.0`, authenticated `weights_only=True` checkpoint loading, relying-party verifier contracts, bounded parser/solver budgets, and SHA-256 supply-chain pins |
| `< 5.2.1` | No | Upgrade to `>= 5.2.1` |

## Trust Boundaries & Security Controls

1. **Authenticated Tensor-Only Checkpoint Loading (`topo_chess.security.checkpoint_io`)**
   - `torch>=2.10.0` is enforced at package constraint and runtime checkpoint boundary (`CVE-2025-32434` fixed in `2.6.0`; `CVE-2026-24747` fixed in `2.10.0`).
   - Checkpoints are read once via `O_RDONLY | O_NOFOLLOW` with a strict byte ceiling (`512 MiB`), verified with `hmac.compare_digest` against a trusted SHA-256 manifest **before** `torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)` runs, and schema-checked before `nn.Module` allocation.
   - Do **not** pass untrusted `.pt` files to `allow_unauthenticated_trusted_local=True`; always provide `--expected-manifest` (`--require-trusted-manifest` in clean-room verification).

2. **Relying-Party Certificate Verification (`topo_brick.evaluation.independent_verifier`)**
   - `AssemblyCertificate` JSON files are untrusted submitter artifacts. When accepting third-party certificates, callers MUST supply explicit `VerificationObligations` (`expected_target_field` + `min_target_iou` or `expected_target_sha256`, `expected_license_filter`, `expected_foundation_policy`, `expected_inventory_counts` / `require_explicit_inventory_spec=True`), a relying-party `MechanicsVerificationProfile`, and `VerifierResourceLimits` (or `verify_certificate_in_isolated_worker(..., timeout_sec=...)`).
   - Never rely on submitter-embedded `inventory_counts`, `foundation_policy`, or inflated static-equilibrium capacities as authorization proof.
   - Local compiled LDraw/Shadow catalogs are verified from raw disk bytes against pinned release SHA-256 digests (`topo_brick.interface.release_provenance`) unless `trusted_custom_catalog_sha256` is explicitly supplied by the relying party.

3. **Bounded Parsers, Subprocesses & Container Isolation**
   - LDraw/Shadow `.dat` file sizes, LDCad `!LDCAD SNAP_INCL` grid expansions, recursive subfile node/triangle/snap budgets, and Lichess CSV line lengths are strictly bounded.
   - `StockfishDistiller` validates an executable regular-file UCI binary, passes a minimal sanitized child environment, and enforces non-blocking selector timeouts and line/byte caps.
   - Container builds pin base image digests (`@sha256:...`), install dependencies with `pip --require-hashes` (`requirements/py313-lock.txt`), lock `/workspace/topographical-tensors` read-only, and execute as non-root `USER 10001:10001`.

## Reporting a Vulnerability

Please report security vulnerabilities privately by emailing `jacobrsullivan@gmail.com` with:
- Affected version and commit SHA,
- Minimal reproduction steps or proof-of-concept input, and
- Expected vs. observed trust-boundary behavior.
