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


class NonlinearDynamicSolver:
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
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | None = None,
        save_interval: int = 1,
        verbose: bool = True,
    ) -> NonlinearDynamicResult:
        """
        Integrates the nonlinear equations of motion forward in time.

        Parameters:
        -----------
        dirichlet : DirichletManager or list of DirichletConstraint
            Essential boundary conditions.
        f_ext : np.ndarray of shape (n_dofs,)
            External load vector applied over time (e.g. constant gravity).
        dt : float
            Time step size in seconds.
        n_steps : int
            Total number of time steps.
        max_iterations : int, default=20
            Maximum Newton-Raphson iterations per time step.
        tolerance : float, default=1e-4
            Relative force residual tolerance.
        include_stresses: bool, default=False
            Calculate collection of postprocessing stresses.
        smooth_to_nodes: bool, default=False
            Convert cell data to point data.
        save_directory : Path, optional
            Directory to export time-series .vtu files and master .pvd animation file.
        save_interval : int, default=1
            Save output every N steps.
        verbose : bool, default=True
            Print step-by-step progress.

        Returns:
        --------
        result : NonlinearDynamicResult
        """
        if isinstance(dirichlet, list):
            manager = DirichletManager(self.domain.mesh.n_dofs)
            for c in dirichlet:
                manager.add_constraint(c)
        else:
            manager = dirichlet

        free_dofs = manager.free_dofs
        p_dofs = manager.prescribed_dofs

        # 1. Assemble Constant Matrices (Mass M and Damping C)
        M = self.domain.assemble_mass_matrix()
        K0 = self.domain.assemble_stiffness_matrix()
        C = self.damping.assemble(M, K0)

        # 2. Slice to Free DOFs
        M_ff = M[free_dofs, :][:, free_dofs].tocsc()
        C_ff = C[free_dofs, :][:, free_dofs].tocsc()
        f_f = f_ext[free_dofs]

        # 3. Newmark-beta Integration Constants
        c0 = 1.0 / (self.beta * dt * dt)
        c1 = self.gamma / (self.beta * dt)
        c2 = 1.0 / (self.beta * dt)
        c3 = (1.0 / (2.0 * self.beta)) - 1.0
        c4 = (self.gamma / self.beta) - 1.0
        c5 = (dt / 2.0) * ((self.gamma / self.beta) - 2.0)

        # 4. State Vectors at start of step n
        u_n = self.domain.u.copy()
        v_n = self.domain.v.copy()
        a_n = self.domain.a.copy()

        # Output preparation
        pvd_entries: list[tuple[float, str]] = []
        exported_files: list[Path] = []
        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)

        time_points = np.linspace(0.0, dt * n_steps, n_steps + 1)

        # Save initial state (t = 0)
        if save_directory is not None:
            filename = f"step_{0:05d}.vtu"
            vtu_path = save_directory / filename
            self.domain.save_state(
                filepath=vtu_path,
                include_stresses=include_stresses,
                smooth_to_nodes=smooth_to_nodes,
            )
            pvd_entries.append((0.0, filename))
            exported_files.append(vtu_path)

        norm_f = max(float(np.linalg.norm(f_f)), 1.0)
        iteration_history: list[int] = []
        converged_all = True

        # 5. Time Integration Loop
        for step in range(1, n_steps + 1):
            t_current = step * dt

            # Initial guess for state at step n+1
            u_k = u_n.copy()
            step_converged = False

            for it in range(1, max_iterations + 1):
                # Update kinematic accelerations and velocities from Newmark formulas
                a_k_free = (
                    c0 * (u_k[free_dofs] - u_n[free_dofs])
                    - c2 * v_n[free_dofs]
                    - c3 * a_n[free_dofs]
                )
                v_k_free = (
                    c1 * (u_k[free_dofs] - u_n[free_dofs])
                    - c4 * v_n[free_dofs]
                    - c5 * a_n[free_dofs]
                )

                # Update domain with current displacement guess
                self.domain.update_state(u=u_k)
                f_int = self.domain.assemble_internal_forces()

                # Dynamic Residual: R = F_ext - M * a - C * v - F_int
                R_free = f_f - M_ff @ a_k_free - C_ff @ v_k_free - f_int[free_dofs]
                res_norm = float(np.linalg.norm(R_free)) / norm_f

                # Check convergence
                if res_norm < tolerance:
                    iteration_history.append(it - 1)
                    step_converged = True
                    # Finalize velocities and accelerations
                    self.domain.v[free_dofs] = v_k_free
                    self.domain.a[free_dofs] = a_k_free
                    break

                # Assemble Tangent Stiffness K_tan(u)
                K_tan = self.domain.assemble_tangent_stiffness_matrix()
                K_tan_ff = K_tan[free_dofs, :][:, free_dofs]

                # Effective dynamic tangent: K_eff = K_tan + c1 * C + c0 * M
                K_eff_ff = (K_tan_ff + c1 * C_ff + c0 * M_ff).tocsc()

                # Solve for displacement increment: K_eff_ff * du_f = R_f
                du_free = self.linear_solver.solve(K_eff_ff, R_free)

                # Backtracking line search (guards against element inversion)
                alpha = 1.0
                line_search_ok = False
                for _ in range(6):
                    u_cand = u_k.copy()
                    u_cand[free_dofs] += alpha * du_free
                    self.domain.update_state(u=u_cand)
                    try:
                        self.domain.assemble_internal_forces()
                        u_k = u_cand
                        line_search_ok = True
                        break
                    except ValueError:
                        alpha *= 0.5

                if not line_search_ok:
                    if verbose:
                        print(
                            f"  Step {step:4d}: Element inversion could not be resolved by line search."
                        )
                    break

            if not step_converged:
                converged_all = False
                if verbose:
                    print(
                        f"[bold red]Nonlinear dynamic solver diverged at step {step}/{n_steps}.[/bold red]"
                    )
                break

            # Advance to next time step: step n becomes step n+1
            u_n = self.domain.u.copy()
            v_n = self.domain.v.copy()
            a_n = self.domain.a.copy()

            # Export snapshot
            if save_directory is not None and (step % save_interval == 0):
                filename = f"step_{step:05d}.vtu"
                vtu_path = save_directory / filename
                self.domain.save_state(
                    filepath=vtu_path,
                    include_stresses=include_stresses,
                    smooth_to_nodes=smooth_to_nodes,
                )
                pvd_entries.append((t_current, filename))
                exported_files.append(vtu_path)

            if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps):
                max_d = np.linalg.norm(self.domain.displacement, axis=1).max()
                print(
                    f"  Step {step:5d}/{n_steps} (t = {t_current:.4f} s) | "
                    f"NR iters: {iteration_history[-1]} | Max Disp = {max_d:.3f} mm"
                )

        # Write master .pvd animation file
        if save_directory is not None and pvd_entries:
            pvd_path = save_directory / "output.pvd"
            _write_pvd_file(pvd_path, pvd_entries)
            if verbose:
                print(
                    f"\n[bold green]ParaView animation written to:[/bold green] {pvd_path}"
                )

        return NonlinearDynamicResult(
            time=time_points,
            converged=converged_all,
            steps_completed=len(iteration_history),
            iteration_history=iteration_history,
            time_series_files=exported_files,
        )


def _write_pvd_file(pvd_path: Path, entries: list[tuple[float, str]]) -> None:
    """Writes a ParaView Data (.pvd) collection file linking the time series."""
    xml_content = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        "  <Collection>",
    ]
    for timestep, filename in entries:
        xml_content.append(
            f'    <DataSet timestep="{timestep:.6e}" group="" part="0" file="{filename}"/>'
        )
    xml_content.append("  </Collection>")
    xml_content.append("</VTKFile>")

    pvd_path.write_text("\n".join(xml_content))
