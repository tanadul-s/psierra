from __future__ import annotations
from typing import Literal
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla

from psierra.solver.base import SparseLinearSolver


class DirectSolver(SparseLinearSolver):
    """
    Direct sparse solver using SuperLU via scipy.sparse.linalg.spsolve.
    Best for general small to medium static problems.
    """

    def solve(self, A: spsp.spmatrix, b: np.ndarray) -> np.ndarray:
        if not isinstance(A, spsp.csc_matrix):
            A = A.tocsc()
        return spspla.spsolve(A, b)


class FactorizedSolver(SparseLinearSolver):
    """
    Pre-factorizes matrix A once, allowing multiple O(N) back-substitutions.
    Ideal for dynamic simulations or multiple load cases.
    """

    def __init__(self, A: spsp.spmatrix) -> None:
        if not isinstance(A, spsp.csc_matrix):
            A = A.tocsc()
        self._solve_fn = spspla.factorized(A)

    def solve(self, A: spsp.spmatrix, b: np.ndarray) -> np.ndarray:
        return self._solve_fn(b)


class ConjugateGradientSolver(SparseLinearSolver):
    """
    Iterative Conjugate Gradient solver for large symmetric positive-definite (SPD) systems.
    Memory-efficient for massive 3D meshes (>100k DOFs).
    """

    def __init__(
        self,
        tolerance: float = 1e-8,
        max_iterations: int = 2000,
        preconditioner: Literal["jacobi"] | None = "jacobi",
    ) -> None:
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.preconditioner = preconditioner

    def solve(self, A: spsp.spmatrix, b: np.ndarray) -> np.ndarray:
        if not isinstance(A, spsp.csr_matrix):
            A = A.tocsr()

        M_prec = None
        if self.preconditioner == "jacobi":
            diag = A.diagonal()
            inv_diag = np.where(np.abs(diag) > 1e-14, 1.0 / diag, 1.0)
            M_prec = spspla.LinearOperator(shape=A.shape, matvec=lambda x: inv_diag * x)

        x, info = spspla.cg(
            A, b, M=M_prec, rtol=self.tolerance, maxiter=self.max_iterations
        )
        if info != 0:
            raise RuntimeError(f"Conjugate Gradient did not converge (info={info}).")
        return x
