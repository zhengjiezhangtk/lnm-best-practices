"""Null models for Lesion Network Mapping (LNM) statistical inference.

This module provides null model implementations for testing the statistical
significance of lesion network mapping results, including spatial, topological,
permutation-based, and spin-test approaches.

References:
    Petersen et al. (2026) - Spatial null models for LNM
    Zalesky et al. (2026) - Topological null models for LNM
    Vos de Wael et al. (2020) - BrainSpace toolbox
"""

from .spatial import SpatialNullModel
from .topological import TopologicalNullModel
from .permutation import PermutationTest
from .spin import SpinTest

__all__ = [
    "SpatialNullModel",
    "TopologicalNullModel",
    "PermutationTest",
    "SpinTest",
]
