"""Representation layer for topo_brick: Topographical Frontier & Candidate Field Generator."""

from topo_brick.representation.interface_fields import (
    CANDIDATE_FEATURE_CHANNELS,
    FRONTIER_PORT_CHANNELS,
    FrontierGraphBatch,
    IntegralTargetGrid,
    SparseTiledTargetField,
    TargetShapeBoxes,
    build_frontier_and_candidate_tensors,
)

__all__ = [
    "FRONTIER_PORT_CHANNELS",
    "CANDIDATE_FEATURE_CHANNELS",
    "TargetShapeBoxes",
    "IntegralTargetGrid",
    "SparseTiledTargetField",
    "FrontierGraphBatch",
    "build_frontier_and_candidate_tensors",
]
