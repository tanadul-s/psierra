from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as spsp
from rich import print

from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.solver.sparse_linear_solver import SparseLinearSolver, DirectSolver


@dataclass
class NonlinearResult:
    """Container for nonlinear static simulation results."""

    displacement: np.ndarray  # Final displacement vector (n_dofs,)
    reactions: np.ndarray  # Support reaction forces (n_dofs,)
    converged: bool  # Whether the solve succeeded
    steps_completed: int  # Number of successful load steps
    iteration_history: list[int] = field(default_factory=list)  # Iterations per step


class NonlinearStaticSolver:
    """
    Nonlinear Static Solver using Incremental Newton-Raphson with Backtracking Line Search.
    Solves R(U) = lambda * F_ext - F_int(U) = 0 using Total Lagrangian kinematics.
    """

    def __init__(
        self,
        domain: SolidDomain,
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
        n_steps: int = 10,
        max_iterations: int = 25,
        tolerance: float = 1e-4,
        verbose: bool = True,
    ) -> NonlinearResult:
        """
        Solves the nonlinear equilibrium equation incrementally.

        Parameters:
        -----------
        dirichlet : DirichletManager or list of DirichletConstraint
            Essential boundary conditions.
        f_ext : np.ndarray of shape (n_dofs,)
            Total external load vector to be reached at lambda = 1.0.
        n_steps : int, default=10
            Number of incremental load steps.
        max_iterations : int, default=25
            Maximum Newton-Raphson iterations per load step.
        tolerance : float, default=1e-4
            Relative force residual convergence tolerance: ||R_f|| / ||F_step|| < tol.
        verbose : bool, default=True
            Whether to print step-by-step convergence progress.

        Returns:
        --------
        result : NonlinearResult
        """
        if isinstance(dirichlet, list):
            manager = DirichletManager(self.domain.mesh.n_dofs)
            for c in dirichlet:
                manager.add_constraint(c)
        else:
            manager = dirichlet

        free_dofs = manager.free_dofs
        p_dofs = manager.prescribed_dofs
        u_p = manager.prescribed_values

        total_dofs = self.domain.mesh.n_dofs
        u_current = self.domain.u.copy()

        iteration_history: list[int] = []
        converged_all = True

        for step in range(1, n_steps + 1):
            lam = step / n_steps
            f_step = lam * f_ext
            f_step_free = f_step[free_dofs]

            # Prescribe displacements proportionally if non-zero
            if len(p_dofs) > 0 and np.any(u_p != 0.0):
                u_current[p_dofs] = lam * u_p

            norm_f = max(float(np.linalg.norm(f_step_free)), 1.0)
            step_converged = False

            for it in range(1, max_iterations + 1):
                # 1. Update state and compute internal forces
                self.domain.update_state(u=u_current)
                f_int = self.domain.assemble_internal_forces()

                # 2. Residual force vector R = F_ext - F_int
                R_free = (f_step - f_int)[free_dofs]
                residual_norm = float(np.linalg.norm(R_free)) / norm_f

                # Check convergence
                if residual_norm < tolerance:
                    iteration_history.append(it - 1)
                    if verbose:
                        print(
                            f"  Step {step:2d}/{n_steps} (λ={lam:.2f}) converged in "
                            f"{it-1} iterations (rel_res = {residual_norm:.2e})"
                        )
                    step_converged = True
                    break

                # 3. Assemble Tangent Stiffness K_tan(u)
                K_tan = self.domain.assemble_tangent_stiffness_matrix()
                K_tan_ff = K_tan[free_dofs, :][:, free_dofs]

                # 4. Solve for displacement increment: K_tan_ff * du_f = R_f
                du_free = self.linear_solver.solve(K_tan_ff, R_free)

                # 5. Backtracking Line Search (Guards against element inversion det(F) <= 0)
                alpha = 1.0
                line_search_success = False

                for _ in range(8):
                    u_candidate = u_current.copy()
                    u_candidate[free_dofs] += alpha * du_free
                    self.domain.update_state(u=u_candidate)

                    try:
                        # Test if deformation gradient det(F) > 0 for all elements
                        self.domain.assemble_internal_forces()
                        u_current = u_candidate
                        line_search_success = True
                        break
                    except ValueError:
                        # Element inversion detected, backtrack step size
                        alpha *= 0.5

                if not line_search_success:
                    if verbose:
                        print(
                            f"  Step {step:2d}/{n_steps}: Line search failed due to element inversion."
                        )
                    step_converged = False
                    break

            if not step_converged:
                converged_all = False
                if verbose:
                    print(
                        f"[bold red]Nonlinear solver diverged at step {step}/{n_steps}.[/bold red]"
                    )
                break

        # Final state update
        self.domain.update_state(u=u_current)

        # Compute reactions: R = F_int - F_ext
        final_f_int = self.domain.assemble_internal_forces()
        reactions = final_f_int - f_ext

        return NonlinearResult(
            displacement=u_current,
            reactions=reactions,
            converged=converged_all,
            steps_completed=len(iteration_history),
            iteration_history=iteration_history,
        )
