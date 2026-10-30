from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spla
from rich import print

from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.damping.rayleigh import RayleighDamping
from psierra.analysis.base import Analysis
from psierra.analysis.exporter import PvdCollectionWriter


@dataclass
class LinearDynamicResult:
    """Container for linear dynamic time-history results."""

    time: np.ndarray  # Time points (n_steps,)
    time_series_files: list[Path] = field(
        default_factory=list
    )  # Exported .vtu file paths


class LinearDynamicAnalysis(Analysis):
    """
    Implicit Linear Dynamic Solver using the Newmark-beta method.
    Solves M * a + C * v + K * u = F(t) over time steps.
    """

    def __init__(
        self,
        domain: SolidDomain,
        damping: RayleighDamping | None = None,
        gamma: float = 0.5,
        beta: float = 0.25,
    ) -> None:
        """
        Parameters:
        -----------
        domain : SolidDomain
            Physical continuum body.
        damping : RayleighDamping, optional
            Rayleigh damping model (defaults to undamped C = 0).
        gamma : float, default=0.5
            Newmark velocity parameter (0.5 ensures second-order accuracy).
        beta : float, default=0.25
            Newmark acceleration parameter (0.25 ensures unconditional numerical stability).
        """
        self.domain = domain
        self.damping = damping if damping is not None else RayleighDamping(0.0, 0.0)
        self.gamma = gamma
        self.beta = beta

    def solve(
        self,
        dirichlet: DirichletManager | list[DirichletConstraint],
        f_ext: np.ndarray,
        dt: float,
        n_steps: int,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_dir: Path | str | None = None,
        output_name: str = "step",
        save_interval: int = 1,
        verbose: bool = True,
    ) -> LinearDynamicResult:
        """
        Executes the linear dynamic time-stepping integration.

        Parameters:
        -----------
        dirichlet : DirichletManager or list of DirichletConstraint
            Essential boundary conditions.
        f_ext : np.ndarray of shape (n_dofs,)
            Constant or peak external force vector.
        dt : float
            Time step size in seconds.
        n_steps : int
            Total number of time steps to compute.
        include_stresses : bool, default=False
            If True, calculates von Mises & principal stresses at each saved step.
        output_dir : Path or str, optional
            Directory to stream Paraview (.vtu) snapshots and master .pvd collection.
        output_name : str, default="step"
            Filename prefix for each exported .vtu frame.
        verbose : bool, default=True
            Print step progress to terminal.

        Returns:
        --------
        result : LinearDynamicResult
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

        # 1. Newmark-beta Integration Constants
        gamma = self.gamma
        beta = self.beta
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * 0.5 * (gamma / beta - 2.0)

        # 2. Assemble Global FEM Matrices (Constant in Linear Dynamics!)
        M = self.domain.assemble_mass_matrix()
        K = self.domain.assemble_stiffness_matrix()
        C = self.damping.assemble_damping_matrix(M, K)

        # 3. Partition System to Free Degrees of Freedom
        M_ff = M[free_dofs, :][:, free_dofs]
        K_ff = K[free_dofs, :][:, free_dofs]
        C_ff = C[free_dofs, :][:, free_dofs]

        f_ext_f = f_ext[free_dofs]

        # 4. Form Effective Tangent Matrix K_eff (Constant!)
        # K_eff = K_ff + a0 * M_ff + a1 * C_ff
        K_eff = (K_ff + a0 * M_ff + a1 * C_ff).tocsc()

        # Factorize K_eff ONCE via SuperLU for blazingly fast O(N) back-substitutions
        solve_step = spla.factorized(K_eff)

        # 5. Initialize State on Free DOFs
        u_f = self.domain.u[free_dofs].copy()
        v_f = self.domain.v[free_dofs].copy()

        # Initial acceleration: M_ff * a0_f = F_ext_f - C_ff * v_f - K_ff * u_f
        r0 = f_ext_f - C_ff @ v_f - K_ff @ u_f
        M_ff_csc = M_ff.tocsc()
        a_f = spla.spsolve(M_ff_csc, r0)

        # Full vectors for output
        u_full = np.zeros(self.domain.mesh.n_dofs, dtype=np.float64)
        v_full = np.zeros(self.domain.mesh.n_dofs, dtype=np.float64)
        a_full = np.zeros(self.domain.mesh.n_dofs, dtype=np.float64)

        if len(p_dofs) > 0:
            u_full[p_dofs] = u_p

        # 6. Time Loop Preparation
        time_points = np.linspace(0, n_steps * dt, n_steps + 1)
        exported_files: list[Path] = []
        pvd_writer = (
            PvdCollectionWriter(save_dir / "output.pvd")
            if save_dir is not None
            else None
        )

        # Save Initial State (Step 0)
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
            print("[bold cyan]Linear Dynamic Analysis (Implicit Newmark-beta)[/bold cyan]")
            print(f"Total DOFs: {self.domain.mesh.n_dofs} | Free DOFs: {len(free_dofs)} | Total Steps: {n_steps} | dt = {dt:.4e} s\n")

        # 7. Time Integration Steps
        for step in range(1, n_steps + 1):
            t_current = step * dt

            # Predictor load: F_eff = F_ext_f + M_ff @ (a0*u + a2*v + a3*a) + C_ff @ (a1*u + a4*v + a5*a)
            m_part = a0 * u_f + a2 * v_f + a3 * a_f
            c_part = a1 * u_f + a4 * v_f + a5 * a_f
            f_eff = f_ext_f + M_ff @ m_part + C_ff @ c_part

            # Solve for next displacement u_{t+dt} via pre-factorized solver
            u_f_next = solve_step(f_eff)

            # Compute next acceleration and velocity using Newmark equations
            a_f_next = a0 * (u_f_next - u_f) - a2 * v_f - a3 * a_f
            v_f_next = v_f + dt * ((1.0 - gamma) * a_f + gamma * a_f_next)

            # Update vectors for next iteration
            u_f = u_f_next
            v_f = v_f_next
            a_f = a_f_next

            # Reconstruct full state
            u_full[free_dofs] = u_f
            v_full[free_dofs] = v_f
            a_full[free_dofs] = a_f

            self.domain.update_state(u=u_full, v=v_full, a=a_full)

            # Save frame if directory specified and matching save_interval
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
                    f"  Step {step:5d}/{n_steps} (t = {t_current:.4f} s) | Max Disp = {max_d:.3f} mm"
                )

        # 8. Write Master .pvd Animation File for ParaView
        if save_dir is not None and pvd_writer is not None:
            pvd_writer.write()

        if verbose:
            print(f"[bold green]✓ Linear Dynamic Analysis completed successfully ({n_steps}/{n_steps} steps).[/bold green]\n")

        return LinearDynamicResult(time=time_points, time_series_files=exported_files)

