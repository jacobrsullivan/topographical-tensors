# Topographical Tensors (`topo_chess` & `topo_brick`) Clean-Room Reproduction Container
FROM python:3.13-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72

LABEL maintainer="Jacob Sullivan <jacobrsullivan@gmail.com>"
LABEL version="5.2.1"
LABEL description="Deterministic clean-room reproduction container for topo_chess (v5.0.0) and topo_brick (Gates 1-9D + Phase 13 security hardening)"

# Deterministic seed & single-core numerical stability environment variables
ENV PYTHONHASHSEED=42 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    STOCKFISH_PATH=/usr/games/stockfish \
    TOPO_DATA_DIR=/data/topographical-tensors \
    TMPDIR=/tmp/topo-scratch

# Install Stockfish UCI engine and create unprivileged runtime account + isolated scratch/data dirs
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 topouser \
    && useradd -u 10001 -g 10001 -m -d /home/topouser -s /usr/sbin/nologin topouser \
    && mkdir -p /data/topographical-tensors /tmp/topo-scratch \
    && chown -R 10001:10001 /data/topographical-tensors /tmp/topo-scratch \
    && chmod 0700 /data/topographical-tensors /tmp/topo-scratch

WORKDIR /workspace/topographical-tensors

# Copy SHA-256 lockfiles first for deterministic layer caching
COPY requirements/ ./requirements/
RUN pip install --no-cache-dir --require-hashes -r requirements/build-lock.txt \
    && pip install --no-cache-dir --require-hashes -r requirements/py313-lock.txt

# Copy package metadata, licenses, source code, and verification ledgers
COPY pyproject.toml README.md SECURITY.md LICENSE DATA_LICENSE.md THIRD_PARTY_NOTICES.md CITATION.cff ./
COPY configs/ ./configs/
COPY docs/ ./docs/
COPY evaluations/ ./evaluations/
COPY logs/ ./logs/
COPY paper/ ./paper/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install package non-editably without network/build-isolation drift, then lock source tree read-only
RUN pip install --no-cache-dir --no-deps --no-build-isolation . \
    && chmod -R a-w /workspace/topographical-tensors

# trivy:ignore:DS-0026 — finite batch verification job; container exit status is the authoritative health signal
USER 10001:10001

# Default entrypoint runs lint + fast verification gates + Phase 13 security hardening audit
CMD ["pytest", "-v", "tests/test_p13_security_hardening_audit.py", "tests/test_gate1_encoding_and_hygiene.py", "tests/test_gate2_ipa_equivariance.py", "tests/test_gate3_d3pm_legality.py", "tests/test_phase_d_audit_compliance.py", "tests/test_phase_e_round2_compliance.py", "tests/test_topo_brick/test_wp_l1_interface.py", "tests/test_topo_brick/test_wp_l2_fields.py", "tests/test_topo_brick/test_wp_l3_planner_and_equilibrium.py", "tests/test_topo_brick/test_revised_construction_kernel.py", "tests/test_topo_brick/test_p12_contract_and_falsification_audit.py", "tests/test_topo_brick/test_p14_tb_fix_22nov_audit.py"]
