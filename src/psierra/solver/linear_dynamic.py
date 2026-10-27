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


@dataclass
class DynamicResult:
    """Container for linear dynamic time-history results."""

    time: np.ndarray  # Time points (n_steps,)
    time_series_files: list[Path] = field(
        default_factory=list
    )  # Exported .vtu file paths


class LinearDynamicSolver:
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
        save_directory: Path | None = None,
        save_interval: int = 1,
        verbose: bool = True,
    ) -> DynamicResult:
        """
        Integrates the equation of motion forward in time.

        Parameters:
        -----------
        dirichlet : DirichletManager or list of DirichletConstraint
            Essential boundary conditions.
        f_ext : np.ndarray of shape (n_dofs,)
            Static external load vector applied over time (e.g. constant gravity).
        dt : float
            Time step size in seconds.
        n_steps : int
            Total number of time steps.
        include_stresses: bool, default=False
            Calculate collection of postprocessing stresses.
        smooth_to_nodes: bool, default=False
            Convert cell data to point data.
        save_directory : Path, optional
            Directory to export time-series .vtu files and the master .pvd animation file.
        save_interval : int, default=1
            Save output every N steps.
        verbose : bool, default=True
            Print step progress.

        Returns:
        --------
        result : DynamicResult
        """
        if isinstance(dirichlet, list):
            manager = DirichletManager(self.domain.mesh.n_dofs)
            for c in dirichlet:
                manager.add_constraint(c)
        else:
            manager = dirichlet

        free_dofs = manager.free_dofs
        p_dofs = manager.prescribed_dofs

        # 1. Assemble Global System Matrices
        K = self.domain.assemble_stiffness_matrix()
        M = self.domain.assemble_mass_matrix()
        C = self.damping.assemble(M, K)

        # 2. Slice to Free DOFs
        K_ff = K[free_dofs, :][:, free_dofs].tocsc()
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

        # 4. Form Effective Dynamic Stiffness Matrix K_eff
        # K_eff = K_ff + c1 * C_ff + c0 * M_ff
        K_eff = (K_ff + c1 * C_ff + c0 * M_ff).tocsc()

        # 5. Factorize K_eff ONCE (Enables O(N) back-substitution for all time steps!)
        solve_step = spla.factorized(K_eff)

        # 6. Initialize State Vectors on Free DOFs
        u_f = self.domain.u[free_dofs].copy()
        v_f = self.domain.v[free_dofs].copy()

        # Initial acceleration: M_ff * a_0 = F_f - C_ff * v_0 - K_ff * u_0
        rhs_a0 = f_f - C_ff @ v_f - K_ff @ u_f
        a_f = spla.spsolve(M_ff, rhs_a0)
        self.domain.a[free_dofs] = a_f

        # Output preparation
        pvd_entries: list[tuple[float, str]] = []
        exported_files: list[Path] = []
        if save_directory is not None:
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)

        time_points = np.linspace(0.0, dt * n_steps, n_steps + 1)

        # Save initial state (step 0)
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

        # 7. Dynamic Time Stepping Loop
        for step in range(1, n_steps + 1):
            t_current = step * dt

            # Predictor RHS:
            # f_eff = f_f + M_ff * (c0 * u_f + c2 * v_f + c3 * a_f) + C_ff * (c1 * u_f + c4 * v_f + c5 * a_f)
            m_part = c0 * u_f + c2 * v_f + c3 * a_f
            c_part = c1 * u_f + c4 * v_f + c5 * a_f
            f_eff = f_f + M_ff @ m_part + C_ff @ c_part

            # Solve for displacement at t + dt in O(N) time!
            u_next_f = solve_step(f_eff)

            # Update acceleration and velocity:
            a_next_f = c0 * (u_next_f - u_f) - c2 * v_f - c3 * a_f
            v_next_f = v_f + dt * ((1.0 - self.gamma) * a_f + self.gamma * a_next_f)

            # Advance state vectors
            u_f, v_f, a_f = u_next_f, v_next_f, a_next_f

            # Update SolidDomain
            self.domain.u[free_dofs] = u_f
            self.domain.v[free_dofs] = v_f
            self.domain.a[free_dofs] = a_f

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
                    f"  Step {step:5d}/{n_steps} (t = {t_current:.4f} s) | Max Disp = {max_d:.3f} mm"
                )

        # 8. Write Master .pvd Animation File for ParaView
        if save_directory is not None and pvd_entries:
            pvd_path = save_directory / "output.pvd"
            _write_pvd_file(pvd_path, pvd_entries)
            if verbose:
                print(
                    f"\n[bold green]ParaView animation written to:[/bold green] {pvd_path}"
                )

        return DynamicResult(time=time_points, time_series_files=exported_files)


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
