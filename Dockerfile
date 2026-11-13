# Topographical Tensors (`topo_chess` & `topo_brick`) Clean-Room Reproduction Container
FROM python:3.13-slim-bookworm

LABEL maintainer="Jacob Sullivan <jacobrsullivan@gmail.com>"
LABEL version="5.1.0"
LABEL description="Deterministic clean-room reproduction container for topo_chess (v5.0.0) and topo_brick (Gates 1-8B)"

# Deterministic seed & single-core numerical stability environment variables
ENV PYTHONHASHSEED=42 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    TOPO_DATA_DIR=/data/topographical-tensors

# Install system dependencies and Stockfish UCI engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/topographical-tensors

# Copy package metadata, licenses, source code, and verification ledgers
COPY pyproject.toml README.md LICENSE DATA_LICENSE.md THIRD_PARTY_NOTICES.md CITATION.cff ./
COPY configs/ ./configs/
COPY docs/ ./docs/
COPY evaluations/ ./evaluations/
COPY logs/ ./logs/
COPY paper/ ./paper/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install package and verification dependencies
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Default entrypoint runs lint + fast verification gates across both topo_chess and topo_brick
CMD ["pytest", "-v", "tests/test_gate1_encoding_and_hygiene.py", "tests/test_gate2_ipa_equivariance.py", "tests/test_gate3_d3pm_legality.py", "tests/test_phase_d_audit_compliance.py", "tests/test_phase_e_round2_compliance.py", "tests/test_topo_brick/test_wp_l1_interface.py", "tests/test_topo_brick/test_wp_l2_fields.py", "tests/test_topo_brick/test_wp_l3_planner_and_equilibrium.py", "tests/test_topo_brick/test_revised_construction_kernel.py"]
