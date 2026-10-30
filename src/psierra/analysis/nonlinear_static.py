from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from rich import print

from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.solver.sparse_linear_solver import SparseLinearSolver, DirectSolver
from psierra.analysis.base import Analysis


@dataclass
class NonlinearStaticResult:
    """Container for nonlinear static simulation results."""

    displacement: np.ndarray  # Final displacement vector (n_dofs,)
    reactions: np.ndarray  # Support reaction forces (n_dofs,)
    converged: bool  # Whether the solve succeeded
    steps_completed: int  # Number of successful load steps
    iteration_history: list[int] = field(default_factory=list)  # Iterations per step


class NonlinearStaticAnalysis(Analysis):
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
    ) -> NonlinearStaticResult:
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
            Relative Euclidean residual convergence tolerance ||R|| / ||F_ext||.
        verbose : bool, default=True
            Print step-by-step residual and convergence reports.

        Returns:
        --------
        result : NonlinearStaticResult
        """
        # Ensure DirichletManager
        if isinstance(dirichlet, list):
            manager = DirichletManager(self.domain.mesh.n_dofs)
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

        # Baseline reference scale for convergence check
        f_ext_free = f_ext[free_dofs]
        f_ref_norm = np.linalg.norm(f_ext_free)
        if f_ref_norm < 1e-12:
            f_ref_norm = 1.0  # Avoid division by zero for unforced problems

        u_current = np.zeros(self.domain.mesh.n_dofs, dtype=np.float64)
        iteration_history: list[int] = []

        if verbose:
            print("[bold cyan]Nonlinear Static Analysis (Incremental Newton-Raphson)[/bold cyan]")
            print(f"Total DOFs: {len(u_current)} | Free DOFs: {len(free_dofs)} | Load Steps: {n_steps} | Max Iter/Step: {max_iterations} | Tol: {tolerance:.1e}\n")

        for step in range(1, n_steps + 1):
            load_factor = step / n_steps
            f_step = load_factor * f_ext
            f_step_free = f_step[free_dofs]

            # Scale prescribed displacements proportionally if non-zero
            if len(p_dofs) > 0 and np.any(u_p != 0.0):
                u_current[p_dofs] = load_factor * u_p

            if verbose:
                print(f"[bold cyan]--- Step {step:2d}/{n_steps} (Load Factor lambda = {load_factor:.2f}) ---[/bold cyan]")

            converged_step = False

            for iteration in range(1, max_iterations + 1):
                # 1. Update state in domain with current displacement guess
                self.domain.update_state(u=u_current)

                # 2. Assemble internal forces F_int(U) and tangent stiffness K_tan(U)
                f_int = self.domain.assemble_internal_forces()
                K_tan = self.domain.assemble_tangent_stiffness_matrix()

                # 3. Compute Out-of-Balance Residual Vector: R = F_ext_step - F_int
                r_free = f_step_free - f_int[free_dofs]
                residual_norm = np.linalg.norm(r_free)
                rel_residual = residual_norm / f_ref_norm

                if verbose:
                    print(f"  Iter {iteration:2d} | ||R|| = {residual_norm:.4e} | Rel = {rel_residual:.4e}")

                # Check Convergence
                if rel_residual < tolerance:
                    converged_step = True
                    iteration_history.append(iteration)
                    if verbose:
                        print(f"  [bold green]✓ Step {step} converged in {iteration} iteration(s)![/bold green]\n")
                    break

                # 4. Partition Tangent Stiffness to Free DOFs
                K_tan_free = K_tan[free_dofs, :][:, free_dofs].tocsc()

                # 5. Solve for Displacement Increment: K_tan_free * du = r_free
                du_free = self.linear_solver.solve(K_tan_free, r_free)

                # 6. Backtracking Line Search (Guards against element inversion det(F) <= 0)
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
                            f"  [bold red]Step {step:2d}/{n_steps}: Line search failed due to element inversion.[/bold red]"
                        )
                    converged_step = False
                    break

            if not converged_step:
                if verbose:
                    print(f"[bold red]✗ Step {step} failed to converge after {max_iterations} iterations![/bold red]")
                self.domain.update_state(u=u_current)
                reactions = self.domain.assemble_internal_forces() - f_step
                return NonlinearStaticResult(
                    displacement=u_current,
                    reactions=reactions,
                    converged=False,
                    steps_completed=step - 1,
                    iteration_history=iteration_history,
                )

        # Simulation successfully completed all steps
        self.domain.update_state(u=u_current)
        f_int_final = self.domain.assemble_internal_forces()
        reactions = f_int_final - f_ext

        if verbose:
            print(f"[bold green]✓ Nonlinear Static Analysis converged successfully ({n_steps}/{n_steps} steps).[/bold green]\n")

        return NonlinearStaticResult(
            displacement=u_current,
            reactions=reactions,
            converged=True,
            steps_completed=n_steps,
            iteration_history=iteration_history,
        )

