from __future__ import annotations
import numpy as np
import scipy.sparse as spsp


class SparseLinearSolver:
    """Base class for sparse linear system solvers (A x = b)."""

    def solve(self, A: spsp.spmatrix, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement solve.")