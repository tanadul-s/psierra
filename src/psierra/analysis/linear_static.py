from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from psierra.domain.base import Domain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.solver.sparse_linear_solver import SparseLinearSolver, DirectSolver
from psierra.analysis.base import Analysis
from rich import print


@dataclass
class LinearStaticResult:
    """Container for linear static simulation results."""

    displacement: np.ndarray  # Full displacement vector (n_dofs,)
    reactions: np.ndarray  # Global reaction force vector (n_dofs,)
    strain_energy: float  # Elastic strain energy (0.5 * u^T K u)
    free_dofs: np.ndarray  # Unconstrained global DOF indices
    prescribed_dofs: np.ndarray  # Constrained global DOF indices


class LinearStaticAnalysis(Analysis):
    """
    Solves the linear static boundary value problem:
    K_ff * u_f = F_f - K_fp * u_p
    """

    def __init__(
        self,
        domain: Domain,
        linear_solver: SparseLinearSolver | None = None,
    ) -> None:
        self.domain = domain
        self.linear_solver = (
            linear_solver if linear_solver is not None else DirectSolver()
        )

    def solve(
        self,
        dirichlet: DirichletManager | list[DirichletConstraint],
        f_ext: np.ndarray,
        verbose: bool = True,
    ) -> LinearStaticResult:
        """
        Solves the static equilibrium equation.

        Parameters:
        -----------
        dirichlet : DirichletManager or list of DirichletConstraint
            Essential boundary conditions.
        f_ext : np.ndarray of shape (n_dofs,)
            Total external load vector (point forces + body gravity loads).
        verbose : bool, default=True
            Whether to print solver progress.

        Returns:
        --------
        result : LinearStaticResult
        """
        # Ensure DirichletManager
        if isinstance(dirichlet, list):
            manager = DirichletManager(self.domain.n_dofs)
            for c in dirichlet:
                manager.add_constraint(c)
        else:
            manager = dirichlet

        free_dofs = manager.free_dofs
        p_dofs = manager.prescribed_dofs
        u_p = manager.prescribed_values

        if len(free_dofs) == 0:
            raise ValueError(
                "All degrees of freedom are constrained. Nothing to solve."
            )

        if verbose:
            print("[bold cyan]Linear Static Analysis (Direct Sparse Factorization)[/bold cyan]")
            print(f"Total DOFs: {self.domain.n_dofs} | Free DOFs: {len(free_dofs)} | Constrained DOFs: {len(p_dofs)}\n")

        # 1. Assemble Global Stiffness Matrix K
        K = self.domain.assemble_stiffness_matrix()

        # 2. Partition Submatrices
        K_ff = K[free_dofs, :][:, free_dofs].tocsc()
        f_f = f_ext[free_dofs]

        # Account for non-zero prescribed displacements: rhs = F_f - K_fp @ u_p
        if len(p_dofs) > 0 and np.any(u_p != 0.0):
            K_fp = K[free_dofs, :][:, p_dofs]
            rhs = f_f - K_fp @ u_p
        else:
            rhs = f_f

        # 3. Solve for Free Displacements: K_ff * u_f = rhs
        u_f = self.linear_solver.solve(K_ff, rhs)

        # 4. Reconstruct Full Displacement Vector U
        u_full = np.zeros(self.domain.n_dofs, dtype=np.float64)
        u_full[free_dofs] = u_f
        if len(p_dofs) > 0:
            u_full[p_dofs] = u_p

        # 5. Compute Reaction Forces: R = K * U - F_ext
        # At free DOFs, R_f ≈ 0. At constrained DOFs, R_p is the support reaction!
        reactions = K @ u_full - f_ext

        # 6. Compute Elastic Strain Energy
        strain_energy = 0.5 * float(u_full @ (K @ u_full))

        # 7. Update SolidDomain State
        self.domain.update_state(u=u_full)

        if verbose:
            print("[bold green]✓ Linear Static Analysis converged successfully.[/bold green]\n")

        return LinearStaticResult(
            displacement=u_full,
            reactions=reactions,
            strain_energy=strain_energy,
            free_dofs=free_dofs,
            prescribed_dofs=p_dofs,
        )
