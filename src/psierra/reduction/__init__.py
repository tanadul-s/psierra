"""
Model Order Reduction (MOR) and Dimension Reduction Framework for Project Sierra.

Provides:
- SnapshotManager: Offline displacement harvesting, .npz auto-caching, and thin SVD.
- PODBasis & SurfacePODBasis: Orthonormal spatial modes and instant surface sub-basis.
- ReducedLinearModel: Projected dense operators (Kr, Mr, Cr, fr) with Dirichlet partitioning.
- ReducedLinearDynamicSolver: Sub-millisecond Generalized-alpha online dynamic solver.
"""

from psierra.reduction.basis import PODBasis, SurfacePODBasis, EnrichedPODBasis
from psierra.reduction.snapshots import SnapshotManager
from psierra.reduction.model import ReducedLinearModel
from psierra.reduction.solver import (
    ReducedLinearDynamicSolver,
    ReducedLinearStaticSolver,
    ReducedSolverStats,
)

__all__ = [
    "PODBasis",
    "SurfacePODBasis",
    "EnrichedPODBasis",
    "SnapshotManager",
    "ReducedLinearModel",
    "ReducedLinearDynamicSolver",
    "ReducedLinearStaticSolver",
    "ReducedSolverStats",
]
