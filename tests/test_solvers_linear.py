from __future__ import annotations
import numpy as np
import pytest
import scipy.sparse as spsp

from psierra.solver.sparse_linear_solver import (
    DirectSolver,
    FactorizedSolver,
    ConjugateGradientSolver,
)


class TestLinearSolvers:
    """Verifies sparse linear equation solvers (A x = b)."""

    @pytest.fixture
    def sample_spd_system(self) -> tuple[spsp.csr_matrix, np.ndarray, np.ndarray]:
        """Creates a well-conditioned 100x100 sparse SPD system with known exact solution."""
        n = 100
        # 1D Poisson matrix: tridiagonal [-1, 2, -1]
        main_diag = np.full(n, 2.0)
        off_diag = np.full(n - 1, -1.0)
        A = spsp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], format="csr")

        x_exact = np.sin(np.linspace(0, np.pi, n))
        b = A @ x_exact
        return A, b, x_exact

    def test_direct_solver(
        self, sample_spd_system: tuple[spsp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        A, b, x_exact = sample_spd_system
        solver = DirectSolver()

        x = solver.solve(A, b)
        assert np.allclose(x, x_exact, atol=1e-12)
        residual = np.linalg.norm(A @ x - b)
        assert residual < 1e-12

    def test_factorized_solver_multiple_rhs(
        self, sample_spd_system: tuple[spsp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        A, b1, x1_exact = sample_spd_system
        solver = FactorizedSolver(A)

        # Solve RHS 1
        x1 = solver.solve(A, b1)
        assert np.allclose(x1, x1_exact, atol=1e-12)

        # Solve RHS 2 with same factorized solver instance
        x2_exact = np.cos(np.linspace(0, np.pi, len(x1_exact)))
        b2 = A @ x2_exact
        x2 = solver.solve(A, b2)
        assert np.allclose(x2, x2_exact, atol=1e-12)

    def test_conjugate_gradient_solver(
        self, sample_spd_system: tuple[spsp.csr_matrix, np.ndarray, np.ndarray]
    ) -> None:
        A, b, x_exact = sample_spd_system
        solver = ConjugateGradientSolver(tolerance=1e-9, max_iterations=1000)

        x = solver.solve(A, b)
        assert np.allclose(x, x_exact, rtol=1e-6, atol=1e-8)
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        assert residual < 1e-8
