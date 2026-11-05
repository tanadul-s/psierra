from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
from rich import print

from psierra.domain.base import Domain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.damping.rayleigh import RayleighDamping
from psierra.analysis.base import Analysis
from psierra.exchange import PvdCollectionWriter


@dataclass
class LinearDynamicResult:
    """Container for linear dynamic time-history results."""

    time: np.ndarray  # Time points (n_steps + 1,)
    time_series_files: list[Path] = field(
        default_factory=list
    )  # Exported .vtu file paths
    converged: bool = True  # True if dynamic integration completed
    steps_completed: int = 0  # Number of completed steps
    steady_state_converged: bool = False  # True if early equilibrium stopping occurred
    steady_state_step: int | None = None  # Step index where steady state was reached


class LinearDynamicAnalysis(Analysis):
    """
    Implicit Linear Dynamic Solver using the Generalized-alpha method (Chung & Hulbert, 1993).
    Solves M * a + C * v + K * u = F(t) over time steps with controllable numerical damping.

    The Generalized-alpha scheme unifies Newmark-beta, HHT-alpha, and Bossak-alpha into a single
    unconditionally stable, strictly second-order accurate O(dt^2) framework controlled via the
    high-frequency spectral radius rho_infinity in [0.0, 1.0].

    Constant prescribed displacements enter through F_f - K_fp u_p, with
    v_p = a_p = 0. Initial free u and v are retained, and a follows the
    reduced equilibrium equation. Sparse factorization is reused each step.
    See docs/06_solvers_and_analysis_pipelines.md and Bathe, Finite Element
    Procedures, ch. 9 (direct integration of the equations of motion).
    """

    def __init__(
        self,
        domain: Domain,
        damping: RayleighDamping | None = None,
        spectral_radius: float = 0.8,
    ) -> None:
        """
        Parameters:
        -----------
        domain : Domain
            Physical continuum or coupled body.
        damping : RayleighDamping, optional
            Rayleigh damping model (defaults to undamped C = 0).
        spectral_radius : float, default=0.8
            High-frequency spectral radius rho_infinity in [0.0, 1.0] controlling algorithmic damping.
            - rho_inf = 1.0: Undamped Newmark-beta / Midpoint Rule (alpha_m = 0.5, alpha_f = 0.5,
              gamma = 0.5, beta = 0.25). Zero numerical dissipation; preserves mechanical energy.
              Best for undamped linear vibrations, resonance tests, and acoustics.
            - rho_inf = 0.8: (Default & Recommended) Optimal high-frequency dissipation for general
              structural dynamics. Selectively suppresses spurious high-frequency spatial mesh noise
              (omega_h ~ c / h_elem) while preserving physical low-frequency structural modes.
            - rho_inf = 0.5: Moderate high-frequency dissipation (equivalent to HHT-alpha with
              alpha_f = 1/3, alpha_m = 0). Ideal for stiff structures with rapid transient shocks.
            - rho_inf = 0.0: Asymptotic annihilation / maximal dissipation (alpha_m = -1.0, alpha_f = 0.0,
              gamma = 1.5, beta = 1.0). Spectral amplification approaches zero in the high-frequency limit.
              Provides strong damping of high-frequency oscillations.
        """
        if not (0.0 <= spectral_radius <= 1.0):
            raise ValueError(
                f"spectral_radius must be in [0.0, 1.0], got {spectral_radius}"
            )

        self.domain = domain
        self.damping = damping if damping is not None else RayleighDamping(0.0, 0.0)
        self.spectral_radius = float(spectral_radius)

        # Generalized-alpha algorithmic constants (Chung & Hulbert, 1993)
        rho = self.spectral_radius
        self.alpha_m = (2.0 * rho - 1.0) / (rho + 1.0)
        self.alpha_f = rho / (rho + 1.0)
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        self.beta = 0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2

    def solve(
        self,
        dirichlet: DirichletManager | list[DirichletConstraint],
        f_ext: np.ndarray,
        dt: float,
        n_steps: int,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_name: str = "result",
        save_interval: int = 1,
        steady_state_tol: float | None = None,
        kinetic_energy_tol: float | None = None,
        consecutive_hold_steps: int = 3,
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
            Nonnegative number of time steps. Zero exports only the initial state.
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
        steady_state_tol : float, optional
            Relative static residual tolerance ||F_reduced - K_ff*u_f|| /
            ||F_reduced||, where F_reduced = F_f - K_fp*u_p. The denominator
            is replaced by 1 for a near-zero reduced load.
            When specified, stops the dynamic solver early once equilibrium is reached.
            If None (default), steady-state checking is ignored.
        kinetic_energy_tol : float, optional
            Relative kinetic energy decay ratio E_k / max(E_k). If provided with
            steady_state_tol, both criteria must be satisfied simultaneously.
        consecutive_hold_steps : int, default=3
            Number of consecutive steps that steady-state criteria must hold before stopping.
        verbose : bool, default=True
            Print step progress to terminal.

        Returns:
        --------
        result : LinearDynamicResult
        """
        self._validate_dynamic_controls(
            dt,
            n_steps,
            save_interval,
            steady_state_tol,
            kinetic_energy_tol,
            consecutive_hold_steps,
        )
        f_ext = np.asarray(f_ext, dtype=np.float64)
        if f_ext.shape != (self.domain.n_dofs,) or not np.all(np.isfinite(f_ext)):
            raise ValueError("f_ext must be a finite vector of shape (n_dofs,)")

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

        save_dir = Path(save_directory) if save_directory is not None else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        hold_counter = 0
        steady_state_converged = False
        steady_state_step: int | None = None
        max_kinetic_energy = 0.0

        # 1. Generalized-alpha Integration Constants
        gamma = self.gamma
        beta = self.beta
        alpha_m = self.alpha_m
        alpha_f = self.alpha_f

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

        # Constant supports have v_p = a_p = 0. Their elastic coupling
        # contributes the reduced load F_f - K_fp u_p (Bathe, ch. 9).
        f_ext_f = f_ext[free_dofs] - K[free_dofs, :][:, p_dofs] @ u_p
        f_ref_norm = np.linalg.norm(f_ext_f)
        if f_ref_norm < 1e-12:
            f_ref_norm = 1.0

        # 4. Form Effective Dynamic Tangent Matrix K_eff (Constant in Linear Dynamics!)
        # K_eff = (1 - alpha_f) * K_ff + (1 - alpha_m) * a0 * M_ff + (1 - alpha_f) * a1 * C_ff
        K_eff = (
            (1.0 - alpha_f) * K_ff
            + (1.0 - alpha_m) * a0 * M_ff
            + (1.0 - alpha_f) * a1 * C_ff
        ).tocsc()

        # Factorize K_eff ONCE via SuperLU for blazingly fast O(N) back-substitutions
        solve_step = spspla.factorized(K_eff)

        # 5. Initialize State on Free DOFs
        u_f = self.domain.u[free_dofs].copy()
        v_f = self.domain.v[free_dofs].copy()

        # Initial acceleration: M_ff * a0_f = F_ext_f - C_ff * v_f - K_ff * u_f
        r0 = f_ext_f - C_ff @ v_f - K_ff @ u_f
        M_ff_csc = M_ff.tocsc()
        a_f = spspla.spsolve(M_ff_csc, r0)

        # Full vectors for output
        u_full = np.zeros(self.domain.n_dofs, dtype=np.float64)
        v_full = np.zeros(self.domain.n_dofs, dtype=np.float64)
        a_full = np.zeros(self.domain.n_dofs, dtype=np.float64)

        if len(p_dofs) > 0:
            u_full[p_dofs] = u_p
        u_full[free_dofs] = u_f
        v_full[free_dofs] = v_f
        a_full[free_dofs] = a_f
        self.domain.update_state(u=u_full, v=v_full, a=a_full)

        # 6. Time Loop Preparation
        time_points = np.linspace(0, n_steps * dt, n_steps + 1)
        exported_files: list[Path] = []
        pvd_writer = (
            PvdCollectionWriter(save_dir / f"{output_name}.pvd")
            if save_dir is not None
            else None
        )

        # Save Initial State (Step 0)
        if save_dir is not None and pvd_writer is not None:
            vtu_name = "step_00000.vtu"
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
            print(
                "[bold cyan]Linear Dynamic Analysis (Implicit Generalized-alpha)[/bold cyan]"
            )
            print(
                f"Total DOFs: {self.domain.n_dofs} | Free DOFs: {len(free_dofs)} | Total Steps: {n_steps} | dt = {dt:.4e} s | rho_inf = {self.spectral_radius:.2f}\n"
            )

        # 7. Time Integration Steps
        for step in range(1, n_steps + 1):
            t_current = step * dt

            # Generalized-alpha RHS effective load vector:
            m_part = (1.0 - alpha_m) * (a0 * u_f + a2 * v_f + a3 * a_f) - alpha_m * a_f
            c_part = (1.0 - alpha_f) * (a1 * u_f + a4 * v_f + a5 * a_f) - alpha_f * v_f
            k_part = -alpha_f * (K_ff @ u_f)
            f_eff = f_ext_f + M_ff @ m_part + C_ff @ c_part + k_part

            # Solve for next displacement u_{t+dt} via pre-factorized solver
            u_f_next = solve_step(f_eff)

            # Compute next acceleration and velocity using Newmark kinematic relations
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
            saved_current_step = False
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
                pvd_writer.add_timestep(t_current, vtu_name)
                exported_files.append(vtu_path)
                saved_current_step = True

            if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps):
                max_d = np.linalg.norm(self.domain.displacement, axis=1).max()
                print(
                    f"  Step {step:5d}/{n_steps} (t = {t_current:.4f} s) | Max Disp = {max_d:.3f} mm"
                )

            # Steady-state equilibrium check (dynamic relaxation / steady state)
            if steady_state_tol is not None:
                r_static = f_ext_f - K_ff @ u_f
                rel_res_static = float(np.linalg.norm(r_static) / f_ref_norm)

                ke_current = 0.5 * float(v_f @ (M_ff @ v_f))
                max_kinetic_energy = max(max_kinetic_energy, ke_current)

                is_equilibrated = rel_res_static <= steady_state_tol
                if kinetic_energy_tol is not None:
                    rel_ke = (
                        ke_current / max_kinetic_energy
                        if max_kinetic_energy > 1e-14
                        else 0.0
                    )
                    is_equilibrated = is_equilibrated and (rel_ke <= kinetic_energy_tol)

                if is_equilibrated:
                    hold_counter += 1
                else:
                    hold_counter = 0

                if hold_counter >= consecutive_hold_steps:
                    steady_state_converged = True
                    steady_state_step = step

                    # Ensure the final snapshot is saved upon reaching steady state
                    if (
                        save_dir is not None
                        and pvd_writer is not None
                        and not saved_current_step
                    ):
                        vtu_name = f"step_{step:05d}.vtu"
                        vtu_path = save_dir / vtu_name
                        self.domain.save_state(
                            vtu_path,
                            include_stresses=include_stresses,
                            smooth_to_nodes=smooth_to_nodes,
                        )
                        pvd_writer.add_timestep(t_current, vtu_name)
                        exported_files.append(vtu_path)

                    if verbose:
                        print(
                            f"\n[bold green]✓ Steady-state static equilibrium reached at step {step} "
                            f"(t = {t_current:.4f} s, Static RelRes = {rel_res_static:.2e})![/bold green]"
                        )
                    break

        # 8. Write Master .pvd Animation File for ParaView
        if save_dir is not None and pvd_writer is not None:
            pvd_writer.write()

        steps_done = steady_state_step if steady_state_converged else n_steps
        time_out = time_points[: steps_done + 1]

        if verbose and not steady_state_converged:
            print(
                f"[bold green]✓ Linear Dynamic Analysis completed successfully ({n_steps}/{n_steps} steps).[/bold green]\n"
            )

        return LinearDynamicResult(
            time=time_out,
            time_series_files=exported_files,
            converged=True,
            steps_completed=steps_done,
            steady_state_converged=steady_state_converged,
            steady_state_step=steady_state_step,
        )
