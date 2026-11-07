# Topographical Tensor Networks & Discrete Diffusion for Chess (v2.0 Clean-Room Container)
FROM python:3.13-slim-bookworm

LABEL maintainer="Topographical Tensors Research Team"
LABEL version="2.0.0"
LABEL description="Deterministic clean-room reproduction container for WP1-WP5 Stage-Gate verification"

# Deterministic seed & numerical stability environment variables
ENV PYTHONHASHSEED=42 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    TOPO_DATA_DIR=/data/topographical-tensors

# Install system dependencies and Stockfish engine for WP1/WP4 fixed-node distillation
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/topographical-tensors

# Copy package metadata and source code
COPY pyproject.toml README.md flight-recorder.md ./
COPY configs/ ./configs/
COPY docs/ ./docs/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Install package and verification dependencies
RUN pip install --upgrade pip && pip install -e ".[dev]"

# Default entrypoint runs the full Stage-Gate 1-5 pytest and clean-room reproduction suite
CMD ["pytest", "tests", "-v"]
