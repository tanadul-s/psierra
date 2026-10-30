from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import scipy.sparse as spsp
from rich import print

from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.damping.rayleigh import RayleighDamping
from psierra.solver.base import SparseLinearSolver
from psierra.solver.sparse_linear_solver import DirectSolver
from psierra.analysis.base import Analysis
from psierra.analysis.exporter import PvdCollectionWriter


@dataclass
class NonlinearDynamicResult:
    """Container for nonlinear dynamic time-history results."""

    time: np.ndarray  # Time points array (n_steps + 1,)
    converged: bool  # True if all time steps converged
    steps_completed: int  # Number of successfully converged steps
    iteration_history: list[int] = field(default_factory=list)  # NR iterations per step
    time_series_files: list[Path] = field(
        default_factory=list
    )  # Exported .vtu file paths


class NonlinearDynamicAnalysis(Analysis):
    """
    Implicit Nonlinear Dynamic Solver using Newmark-beta with Newton-Raphson.
    Solves M * a + C * v + F_int(u) = F_ext(t) for finite deformations.
    """

    def __init__(
        self,
        domain: SolidDomain,
        damping: RayleighDamping | None = None,
        linear_solver: SparseLinearSolver | None = None,
        gamma: float = 0.5,
        beta: float = 0.25,
    ) -> None:
        self.domain = domain
        self.damping = damping if damping is not None else RayleighDamping(0.0, 0.0)
        self.linear_solver = (
            linear_solver if linear_solver is not None else DirectSolver()
        )
        self.gamma = gamma
        self.beta = beta

    def solve(
        self,
        dirichlet: DirichletManager | list[DirichletConstraint],
        f_ext: np.ndarray,
        dt: float,
        n_steps: int,
        max_iterations: int = 15,
        tolerance: float = 1e-4,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_dir: Path | str | None = None,
        output_name: str = "step",
        save_interval: int = 1,
        verbose: bool = True,
    ) -> NonlinearDynamicResult:
        """
        Executes the nonlinear dynamic time-stepping integration.

        Parameters:
        -----------
        dirichlet : DirichletManager or list of DirichletConstraint
            Essential boundary conditions.
        f_ext : np.ndarray of shape (n_dofs,)
            Total external load vector (gravity body forces, boundary loads).
        dt : float
            Time increment in seconds.
        n_steps : int
            Number of time steps to compute.
        max_iterations : int, default=15
            Maximum Newton-Raphson iterations per time step.
        tolerance : float, default=1e-4
            Relative dynamic residual tolerance ||R_eff|| / ||F_ext||.
        include_stresses : bool, default=False
            If True, calculates von Mises & principal stresses at each saved step.
        output_dir : Path or str, optional
            Directory to stream Paraview (.vtu) snapshots and master .pvd collection.
        output_name : str, default="step"
            Filename prefix for each exported .vtu frame.
        verbose : bool, default=True
            Print time step progress and Newton-Raphson residual history.

        Returns:
        --------
        result : NonlinearDynamicResult
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

        target_dir = save_directory if save_directory is not None else output_dir
        save_dir = Path(target_dir) if target_dir is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Newmark Integration Constants
        gamma = self.gamma
        beta = self.beta
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * 0.5 * (gamma / beta - 2.0)

        # 2. Assemble Constant Mass Matrix M and Initial Damping Matrix C
        M = self.domain.assemble_mass_matrix()
        K0 = self.domain.assemble_stiffness_matrix()
        C = self.damping.assemble_damping_matrix(M, K0)

        # Partition mass and damping to free DOFs
        M_ff = M[free_dofs, :][:, free_dofs]
        C_ff = C[free_dofs, :][:, free_dofs]
        f_ext_f = f_ext[free_dofs]

        f_ref_norm = np.linalg.norm(f_ext_f)
        if f_ref_norm < 1e-12:
            f_ref_norm = 1.0

        # 3. Initialize Kinematic Vectors on Free DOFs
        u_f = self.domain.u[free_dofs].copy()
        v_f = self.domain.v[free_dofs].copy()

        # Initial acceleration: M_ff * a0_f = F_ext_f - C_ff * v_f - F_int_0
        f_int0 = self.domain.assemble_internal_forces()
        r0 = f_ext_f - C_ff @ v_f - f_int0[free_dofs]
        a_f = self.linear_solver.solve(M_ff.tocsc(), r0)

        # Full vectors for output
        u_full = np.zeros(self.domain.mesh.n_dofs, dtype=np.float64)
        v_full = np.zeros(self.domain.mesh.n_dofs, dtype=np.float64)
        a_full = np.zeros(self.domain.mesh.n_dofs, dtype=np.float64)

        if len(p_dofs) > 0:
            u_full[p_dofs] = u_p

        # 4. Prepare Time Loop Tracking
        time_points = np.linspace(0, n_steps * dt, n_steps + 1)
        exported_files: list[Path] = []
        iteration_history: list[int] = []
        pvd_writer = (
            PvdCollectionWriter(save_dir / "output.pvd")
            if save_dir is not None
            else None
        )

        # Save Initial Step 0
        if save_dir is not None and pvd_writer is not None:
            vtu_name = f"{output_name}_00000.vtu"
            vtu_path = save_dir / vtu_name
            self.domain.update_state(u=u_full, v=v_full, a=a_full)
            self.domain.save_state(
                vtu_path,
                include_stresses=include_stresses,
                smooth_to_nodes=smooth_to_nodes,
            )
            pvd_writer.add_timestep(0.0, vtu_name)
            exported_files.append(vtu_path)

        if verbose:
            print("[bold cyan]Nonlinear Dynamic Analysis (Implicit Newmark-beta)[/bold cyan]")
            print(
                f"Total DOFs: {self.domain.mesh.n_dofs} | Free DOFs: {len(free_dofs)} | Total Steps: {n_steps} | dt = {dt:.4e} s | Tol = {tolerance:.1e}\n"
            )

        # 5. Dynamic Time Stepping Loop
        for step in range(1, n_steps + 1):
            t_current = step * dt

            # Initial trial state for step n+1 starts at previous displacement u_n
            u_f_i = u_f.copy()
            step_converged = False

            # Newton-Raphson equilibrium iterations within the time step
            for iteration in range(1, max_iterations + 1):
                # Kinematic accelerations and velocities from Newmark relations
                a_f_i = (
                    a0 * (u_f_i - u_f)
                    - a2 * v_f
                    - a3 * a_f
                )
                v_f_i = (
                    a1 * (u_f_i - u_f)
                    - a4 * v_f
                    - a5 * a_f
                )

                # Full vectors for internal force evaluation
                u_full[free_dofs] = u_f_i
                v_full[free_dofs] = v_f_i
                a_full[free_dofs] = a_f_i
                self.domain.update_state(u=u_full, v=v_full, a=a_full)

                # Assemble nonlinear internal forces F_int(u)
                f_int = self.domain.assemble_internal_forces()

                # Dynamic out-of-balance residual: R = F_ext - M*a - C*v - F_int
                inertia_f = M_ff @ a_f_i
                damping_f = C_ff @ v_f_i
                r_eff = f_ext_f - inertia_f - damping_f - f_int[free_dofs]

                res_norm = np.linalg.norm(r_eff)
                rel_res = res_norm / f_ref_norm

                if rel_res < tolerance:
                    step_converged = True
                    iteration_history.append(iteration - 1)
                    break

                # Assemble Tangent Stiffness K_tan(u) and Effective Dynamic Tangent Matrix
                K_tan = self.domain.assemble_tangent_stiffness_matrix()
                K_tan_ff = K_tan[free_dofs, :][:, free_dofs]
                K_eff = (K_tan_ff + a0 * M_ff + a1 * C_ff).tocsc()

                # Solve for displacement increment: K_eff * du = R_eff
                du_f = self.linear_solver.solve(K_eff, r_eff)

                # Backtracking Line Search (Guards against element inversion det(F) <= 0)
                alpha = 1.0
                line_search_ok = False
                for _ in range(8):
                    u_cand = u_f_i + alpha * du_f
                    u_full[free_dofs] = u_cand
                    self.domain.update_state(u=u_full)
                    try:
                        self.domain.assemble_internal_forces()
                        u_f_i = u_cand
                        line_search_ok = True
                        break
                    except ValueError:
                        alpha *= 0.5

                if not line_search_ok:
                    if verbose:
                        print(
                            f"  [bold red]Step {step:4d}: Element inversion could not be resolved by line search.[/bold red]"
                        )
                    break

            if not step_converged:
                if verbose:
                    print(
                        f"[bold red]✗ Step {step} (t={t_current:.4f}s) failed to converge in {max_iterations} iterations![/bold red]"
                    )
                return NonlinearDynamicResult(
                    time=time_points[:step],
                    converged=False,
                    steps_completed=step - 1,
                    iteration_history=iteration_history,
                    time_series_files=exported_files,
                )

            # Accept converged step: advance state
            u_f = u_f_i
            v_f = v_f_i
            a_f = a_f_i

            u_full[free_dofs] = u_f
            v_full[free_dofs] = v_f
            a_full[free_dofs] = a_f
            self.domain.update_state(u=u_full, v=v_full, a=a_full)

            # Save frame
            if (
                save_dir is not None
                and pvd_writer is not None
                and (step % save_interval == 0 or step == n_steps)
            ):
                vtu_name = f"{output_name}_{step:05d}.vtu"
                vtu_path = save_dir / vtu_name
                self.domain.save_state(
                    vtu_path,
                    include_stresses=include_stresses,
                    smooth_to_nodes=smooth_to_nodes,
                )
                pvd_writer.add_timestep(t_current, vtu_name)
                exported_files.append(vtu_path)

            if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps):
                max_d = np.linalg.norm(self.domain.displacement, axis=1).max()
                print(
                    f"  Step {step:5d}/{n_steps} (t = {t_current:.4f} s) | Iter = {iteration_history[-1]:2d} | Max Disp = {max_d:.3f} mm"
                )

        # 6. Write Master .pvd Collection
        if save_dir is not None and pvd_writer is not None:
            pvd_writer.write()

        if verbose:
            print(f"[bold green]✓ Nonlinear Dynamic Analysis completed successfully ({n_steps}/{n_steps} steps).[/bold green]\n")

        return NonlinearDynamicResult(
            time=time_points,
            converged=True,
            steps_completed=n_steps,
            iteration_history=iteration_history,
            time_series_files=exported_files,
        )

