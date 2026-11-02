from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from rich import print

from psierra.domain.base import Domain
from psierra.boundary import DirichletManager, DirichletConstraint, DirichletBoundaryCondition
from psierra.solver.sparse_linear_solver import SparseLinearSolver, DirectSolver
from psierra.analysis.base import Analysis
from psierra.exchange import PvdCollectionWriter


@dataclass
class NonlinearStaticResult:
    """Container for nonlinear static simulation results."""

    displacement: np.ndarray  # Final displacement vector (n_dofs,)
    reactions: np.ndarray  # Support reaction forces (n_dofs,)
    converged: bool  # Whether the solve succeeded
    steps_completed: int  # Number of successful load steps
    iteration_history: list[int] = field(default_factory=list)  # Iterations per step
    time_series_files: list[Path] = field(
        default_factory=list
    )  # Exported .vtu file paths


class NonlinearStaticAnalysis(Analysis):
    """
    Nonlinear Static Solver using Incremental Newton-Raphson with Backtracking Line Search.
    Solves R(U) = lambda * F_ext - F_int(U) = 0 using Total Lagrangian kinematics.
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
        dirichlet: (
            DirichletManager
            | list[DirichletBoundaryCondition]
            | Callable[[int, int], DirichletManager | list[DirichletBoundaryCondition]]
        ),
        f_ext: np.ndarray | None = None,
        n_steps: int = 10,
        nr_iterations: int = 25,
        nr_tol: float = 1e-4,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_name: str = "result",
        save_interval: int = 1,
        verbose: bool = True,
    ) -> NonlinearStaticResult:
        """
        Solves the nonlinear equilibrium equation incrementally.

        Parameters:
        -----------
        dirichlet : DirichletManager, list of DirichletBoundaryCondition, or callable (step, n_steps) -> DirichletManager
            Essential boundary conditions. Can be a static constraint manager or a step-dependent callable.
        f_ext : np.ndarray of shape (n_dofs,), optional
            Total external load vector to be reached at lambda = 1.0. Defaults to zero vector for displacement control.
        n_steps : int, default=10
            Number of incremental load steps.
        nr_iterations : int, default=25
            Maximum Newton-Raphson iterations per load step.
        nr_tol : float, default=1e-4
            Relative Euclidean residual convergence tolerance ||R|| / ||F_ext|| (or relative to step unbalance).
        include_stresses : bool, default=False
            If True, calculates von Mises & principal stresses at each saved step.
        smooth_to_nodes : bool, default=False
            If True, smooths element stresses to mesh nodes.
        save_directory : Path or str, optional
            Directory to stream Paraview (.vtu) snapshots and master .pvd collection.
        output_name : str, default="result"
            Master filename prefix for exported .pvd collection (result.pvd).
        save_interval : int, default=1
            Number of steps between saved step_*.vtu frame files.
        verbose : bool, default=True
            Print step-by-step residual and convergence reports.

        Returns:
        --------
        result : NonlinearStaticResult
        """
        if f_ext is None:
            f_ext = np.zeros(self.domain.n_dofs, dtype=np.float64)

        # Initialize DirichletManager
        if callable(dirichlet):
            initial_manager = dirichlet(1, n_steps)
            if isinstance(initial_manager, list):
                mgr = DirichletManager(self.domain.n_dofs)
                for c in initial_manager:
                    mgr.add_constraint(c)
                initial_manager = mgr
            manager = initial_manager
        elif isinstance(dirichlet, list):
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

        save_dir = Path(save_directory) if save_directory is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        if save_interval < 1:
            raise ValueError(f"save_interval must be >= 1, got {save_interval}")

        u_current = np.zeros(self.domain.n_dofs, dtype=np.float64)
        iteration_history: list[int] = []
        exported_files: list[Path] = []
        pvd_writer = (
            PvdCollectionWriter(save_dir / f"{output_name}.pvd")
            if save_dir is not None
            else None
        )

        # Save Step 0 (Initial state at lambda = 0.0)
        if save_dir is not None and pvd_writer is not None:
            vtu_name = "step_00000.vtu"
            vtu_path = save_dir / vtu_name
            self.domain.update_state(u=u_current)
            self.domain.save_state(
                vtu_path,
                include_stresses=include_stresses,
                smooth_to_nodes=smooth_to_nodes,
            )
            pvd_writer.add_timestep(0.0, vtu_name)
            exported_files.append(vtu_path)

        if verbose:
            print(
                "[bold cyan]Nonlinear Static Analysis (Incremental Newton-Raphson)[/bold cyan]"
            )
            print(
                f"Total DOFs: {len(u_current)} | Free DOFs: {len(free_dofs)} | Load Steps: {n_steps} | Max Iter/Step: {nr_iterations} | Tol: {nr_tol:.1e}\n"
            )

        for step in range(1, n_steps + 1):
            load_factor = step / n_steps

            # Update Dirichlet constraints for the current step
            if callable(dirichlet):
                step_manager = dirichlet(step, n_steps)
                if isinstance(step_manager, list):
                    mgr = DirichletManager(self.domain.n_dofs)
                    for c in step_manager:
                        mgr.add_constraint(c)
                    step_manager = mgr
                free_dofs = step_manager.free_dofs
                p_dofs = step_manager.prescribed_dofs
                u_p = step_manager.prescribed_values
                u_current[p_dofs] = u_p
            elif len(p_dofs) > 0 and np.any(u_p != 0.0):
                u_current[p_dofs] = load_factor * u_p

            f_step = load_factor * f_ext
            f_step_free = f_step[free_dofs]
            f_ext_free_norm = float(np.linalg.norm(f_step_free))
            f_ref_step: float | None = f_ext_free_norm if f_ext_free_norm >= 1e-12 else None

            if verbose:
                print(
                    f"[bold cyan]--- Step {step:2d}/{n_steps} (Load Factor lambda = {load_factor:.2f}) ---[/bold cyan]"
                )

            converged_step = False

            for iteration in range(1, nr_iterations + 1):
                # 1. Update state in domain with current displacement guess
                self.domain.update_state(u=u_current)

                # 2. Assemble internal forces F_int(U) and tangent stiffness K_tan(U)
                f_int = self.domain.assemble_internal_forces()
                K_tan = self.domain.assemble_tangent_stiffness_matrix()

                # 3. Compute Out-of-Balance Residual Vector: R = F_ext_step - F_int
                r_free = f_step_free - f_int[free_dofs]
                residual_norm = float(np.linalg.norm(r_free))

                if f_ref_step is None:
                    # Displacement-controlled problem without external loads:
                    # normalize against the step's initial unbalance or internal force scale
                    f_ref_step = max(residual_norm, float(np.linalg.norm(f_int[free_dofs])), 1.0)

                rel_residual = residual_norm / f_ref_step

                if verbose:
                    print(
                        f"  Iter {iteration:2d} | ||R|| = {residual_norm:.4e} | Rel = {rel_residual:.4e}"
                    )

                # Check Convergence
                if rel_residual < nr_tol:
                    converged_step = True
                    iteration_history.append(iteration)
                    if verbose:
                        print(
                            f"  [bold green]✓ Step {step} converged in {iteration} iteration(s)![/bold green]\n"
                        )
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
                    print(
                        f"[bold red]✗ Step {step} failed to converge after {nr_iterations} iterations![/bold red]"
                    )
                self.domain.update_state(u=u_current)
                if save_dir is not None and pvd_writer is not None:
                    pvd_writer.write()
                reactions = self.domain.assemble_internal_forces() - f_step
                return NonlinearStaticResult(
                    displacement=u_current,
                    reactions=reactions,
                    converged=False,
                    steps_completed=step - 1,
                    iteration_history=iteration_history,
                    time_series_files=exported_files,
                )

            # Save frame if directory specified and matching save_interval or final step
            if (
                save_dir is not None
                and pvd_writer is not None
                and (step % save_interval == 0 or step == n_steps)
            ):
                vtu_name = f"step_{step:05d}.vtu"
                vtu_path = save_dir / vtu_name
                self.domain.save_state(
                    vtu_path,
                    include_stresses=include_stresses,
                    smooth_to_nodes=smooth_to_nodes,
                )
                pvd_writer.add_timestep(float(load_factor), vtu_name)
                exported_files.append(vtu_path)

        # Simulation successfully completed all steps
        self.domain.update_state(u=u_current)
        f_int_final = self.domain.assemble_internal_forces()
        reactions = f_int_final - f_ext

        if save_dir is not None and pvd_writer is not None:
            pvd_writer.write()

        if verbose:
            print(
                f"[bold green]✓ Nonlinear Static Analysis converged successfully ({n_steps}/{n_steps} steps).[/bold green]\n"
            )

        return NonlinearStaticResult(
            displacement=u_current,
            reactions=reactions,
            converged=True,
            steps_completed=n_steps,
            iteration_history=iteration_history,
            time_series_files=exported_files,
        )
