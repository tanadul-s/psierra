from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import time
import numpy as np
import scipy.sparse as spsp
from rich import print

from psierra.domain.base import Domain
from psierra.boundary import (
    DirichletManager,
    DirichletConstraint,
    DirichletBoundaryCondition,
    RigidContact,
    DeformableContact,
    SelfContact,
    MortarContact,
)
from psierra.boundary.contact import ContactMultiplierState
from psierra.solver.sparse_linear_solver import SparseLinearSolver, DirectSolver
from psierra.analysis.base import Analysis
from psierra.analysis.augmentation import ContactAugmentation
from psierra.analysis.line_search import ContactAwareLineSearch, LineSearchResult
from psierra.exchange.writer import SimulationWriter


@dataclass
class QuasiStaticResult:
    """Container for nonlinear quasi-static simulation results."""

    displacement: np.ndarray  # Final displacement vector (n_dofs,)
    reactions: np.ndarray  # Support reaction forces (n_dofs,)
    converged: bool  # Whether the solve succeeded
    steps_completed: int  # Number of successful time steps
    time_history: list[float] = field(default_factory=list)  # Physical time history
    displacement_history: list[np.ndarray] = field(default_factory=list)  # Displacements per step
    iteration_history: list[int] = field(default_factory=list)  # Iterations per step
    time_series_files: list[Path] = field(default_factory=list)  # Exported files
    saved_file: Path | None = None  # Saved master visualization file (e.g. result.xdmf)
    al_iterations_history: list[int] = field(default_factory=list)
    wall_time: float = 0.0


class QuasiStaticAnalysis(Analysis):
    """
    Nonlinear Quasi-Static Analysis Solver with Physical Time Marching and Rate-Dependent Continua.

    Solves the incremental mechanical equilibrium equations over physical time t in [0, T]
    without inertia:
        R(u_{n+1}) = f_ext(t_{n+1}) + f_contact(u_{n+1}) - f_int(u_{n+1}, Q_{n+1}(dt)) = 0

    Governing Principles:
    ---------------------
    - Physical Time Evolution: Steps through physical time increments dt = t_{n+1} - t_n,
      enabling time-dependent viscoelastic relaxation, creep, and rate-dependent dissipation.
    - Transactional Material History: Calls domain.set_time_step(dt), domain.commit_state()
      upon convergence, and domain.revert_state() upon cutback or Newton rejection.
    - Adaptive Time Stepping: Optional automatic cutback (dt <- dt * gamma_cut) if Newton-Raphson
      stalls or diverges, restoring the accepted history state and resuming smoothly.
    - Contact Mechanics: Full integration with Augmented Lagrangian Uzawa iterations,
      rebound relaxation, and line search guards against element inversion.
    - Visualization: Directly streams time snapshots to SimulationWriter (XDMF/PVD) with
      exact physical timestamps.
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
            | Callable[[int, float, float], DirichletManager | list[DirichletBoundaryCondition]]
        ),
        f_ext: np.ndarray | Callable[[float], np.ndarray] | None = None,
        total_time: float = 1.0,
        dt: float = 0.05,
        time_steps: np.ndarray | Sequence[float] | None = None,
        load_schedule: Callable[[float], float] | None = None,
        contact: (
            ContactMultiplierState
            | Sequence[ContactMultiplierState]
            | None
        ) = None,
        nr_iterations: int = 25,
        nr_tol: float = 1e-4,
        adaptive_stepping: bool = False,
        dt_min: float = 1e-5,
        dt_max: float | None = None,
        gamma_cut: float = 0.5,
        gamma_grow: float = 1.2,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_name: str = "result",
        format: str = "xdmf",
        save_interval: int = 1,
        save_history: bool = True,
        verbose: bool = True,
        writer: SimulationWriter | None = None,
        initial_time: float = 0.0,
        *,
        al_max_iterations: int = 50,
        al_gap_tol: float = 1e-6,
        al_tol: float = 1e-6,
        rebound_relaxation: bool = False,
        line_search: ContactAwareLineSearch | None = None,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
    ) -> QuasiStaticResult:
        """
        Solves the nonlinear quasi-static problem across physical time.

        Parameters:
        -----------
        dirichlet : DirichletManager, list of DirichletBoundaryCondition, or callable (step, t, dt)
            Essential boundary conditions.
        f_ext : np.ndarray or callable (t) -> np.ndarray, optional
            External load vector or time-dependent callable.
        total_time : float, default=1.0
            Total simulation physical duration in seconds.
        dt : float, default=0.05
            Default physical time increment in seconds.
        time_steps : Sequence of float, optional
            Explicit list of time steps dt to execute sequentially.
        load_schedule : callable (t) -> float, optional
            Scalar load multiplier function applied to static f_ext.
        contact : Contact conditions, optional
        nr_iterations : int, default=25
            Maximum Newton-Raphson iterations per step.
        nr_tol : float, default=1e-4
            Convergence tolerance for relative residual ||R_free|| / max(||f_ref||, 1.0).
        adaptive_stepping : bool, default=False
            Enable automatic time step cutback on divergence.
        dt_min : float, default=1e-5
            Minimum allowable time step under adaptive cutback.
        dt_max : float, optional
            Maximum allowable time step under adaptive growth.
        save_directory : Path or str, optional
            Output folder for visualization collections.
        format : {'xdmf', 'pvd'}, default='xdmf'
        save_interval : int, default=1
        save_history : bool, default=True
        verbose : bool, default=True
        writer : SimulationWriter, optional
        initial_time : float, default=0.0
        al_max_iterations : int, default=50
        al_gap_tol : float, default=1e-6
        al_tol : float, default=1e-6
        rebound_relaxation : bool, default=False
        line_search : ContactAwareLineSearch, optional
        callback : callable (step, t, u) -> None, optional

        Returns:
        --------
        QuasiStaticResult
        """
        t0_wall = time.perf_counter()

        # 1. Contact augmentation setup
        contact_list: list[ContactMultiplierState] = []
        if contact is not None:
            if isinstance(contact, (list, tuple)):
                contact_list = list(contact)
            else:
                contact_list = [contact]
        augmentation = ContactAugmentation(
            contacts=contact_list,
            max_iterations=al_max_iterations,
            gap_tol=al_gap_tol,
            tolerance=al_tol,
            rebound_relaxation=rebound_relaxation,
        )
        line_search_solver = (
            line_search if line_search is not None else ContactAwareLineSearch()
        )

        # 2. Dirichlet setup
        p_dofs = np.array([], dtype=np.int64)
        u_p = np.array([], dtype=np.float64)
        if not callable(dirichlet):
            if isinstance(dirichlet, list):
                mgr = DirichletManager(self.domain.n_dofs)
                for c in dirichlet:
                    mgr.add_constraint(c)
                dirichlet = mgr
            free_dofs = dirichlet.free_dofs
            p_dofs = dirichlet.prescribed_dofs
            u_p = dirichlet.prescribed_values
        else:
            free_dofs = np.arange(self.domain.n_dofs, dtype=np.int64)

        # 3. Base External Load vector
        if f_ext is None:
            f_ext_base = np.zeros(self.domain.n_dofs, dtype=np.float64)
        elif not callable(f_ext):
            f_ext_base = np.asarray(f_ext, dtype=np.float64).ravel()
            if len(f_ext_base) != self.domain.n_dofs:
                raise ValueError(
                    f"External load length ({len(f_ext_base)}) must match DOFs ({self.domain.n_dofs})"
                )
        else:
            f_ext_base = np.zeros(self.domain.n_dofs, dtype=np.float64)

        # 4. Simulation Writer Initialization
        created_internal_writer = False
        save_dir = Path(save_directory) if save_directory is not None else None
        if save_history and save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            if writer is None:
                target_filename = (
                    output_name
                    if Path(output_name).suffix in (".pvd", ".xdmf", ".h5", ".vtu")
                    else f"{output_name}.{format}"
                )
                writer = SimulationWriter(
                    output_path=save_dir / target_filename,
                    mesh=getattr(self.domain, "mesh", None),
                    format=format,  # type: ignore[arg-type]
                    store_checkpoints=False,
                )
                created_internal_writer = True

            if writer is not None:
                writer.add_timestep(time=initial_time, domain=self.domain, step_index=0)

        # 5. Build time step list
        if time_steps is not None:
            target_dts = [float(s) for s in time_steps]
            total_duration = float(sum(target_dts))
        else:
            n_planned_steps = max(1, int(round(total_time / dt)))
            target_dts = [float(dt)] * n_planned_steps
            total_duration = float(total_time)

        max_dt_allowed = float(dt_max) if dt_max is not None else 2.0 * float(dt)

        # Working state
        u_current = self.domain.u.copy()
        current_time = float(initial_time)
        steps_completed = 0
        iteration_history: list[int] = []
        al_iterations_history: list[int] = []
        time_history: list[float] = [current_time]
        displacement_history: list[np.ndarray] = [u_current.copy()]

        if verbose:
            print("[bold cyan]Quasi-Static Analysis (Rate-Dependent Equilibrium)[/bold cyan]")
            print(
                f"Total DOFs: {len(u_current)} | Free DOFs: {len(free_dofs)} | "
                f"Total Time: {total_duration:.3f} s | Planned Steps: {len(target_dts)} | Tol: {nr_tol:.1e}\n"
            )

        step_idx = 0
        all_converged = True

        while current_time < total_duration - 1e-12 and step_idx < len(target_dts):
            dt_step = target_dts[step_idx]
            step_converged = False

            while not step_converged:
                t_target = current_time + dt_step

                # Set material time step
                if hasattr(self.domain, "set_time_step"):
                    self.domain.set_time_step(dt_step)

                accepted_u = self.domain.u.copy()
                accepted_v = self.domain.v.copy()
                accepted_a = self.domain.a.copy()
                if augmentation.rebound_relaxation:
                    augmentation.relax_rebound()
                multiplier_snapshot = augmentation.snapshot(self.domain.current_nodes)

                # Update Dirichlet constraints
                if callable(dirichlet):
                    step_manager = dirichlet(steps_completed + 1, t_target, dt_step)
                    if isinstance(step_manager, list):
                        mgr = DirichletManager(self.domain.n_dofs)
                        for c in step_manager:
                            mgr.add_constraint(c)
                        step_manager = mgr
                    free_dofs = step_manager.free_dofs
                    p_dofs = step_manager.prescribed_dofs
                    u_p = step_manager.prescribed_values
                    u_current[p_dofs] = u_p
                elif len(p_dofs) > 0:
                    if load_schedule is not None:
                        u_current[p_dofs] = float(load_schedule(t_target)) * u_p
                    else:
                        u_current[p_dofs] = u_p

                # Compute external load for this step
                if callable(f_ext):
                    f_step = f_ext(t_target)
                elif load_schedule is not None:
                    f_step = float(load_schedule(t_target)) * f_ext_base
                else:
                    f_step = f_ext_base.copy()

                f_step_free = f_step[free_dofs]
                f_ext_free_norm = float(np.linalg.norm(f_step_free))
                f_ref_step = f_ext_free_norm if f_ext_free_norm >= 1e-12 else None

                if verbose:
                    print(
                        f"[bold cyan]--- Step {steps_completed + 1} (t = {t_target:.4f} s, dt = {dt_step:.4e} s) ---[/bold cyan]"
                    )

                inner_converged = False
                total_nr_iters = 0

                for al_iteration in range(1, augmentation.max_iterations + 1):
                    for iteration in range(1, nr_iterations + 1):
                        self.domain.update_state(u=u_current)

                        try:
                            f_int = self.domain.assemble_internal_forces()
                            K_tan = self.domain.assemble_tangent_stiffness_matrix()

                            f_contact = np.zeros(self.domain.n_dofs, dtype=np.float64)
                            K_contact = spsp.csr_matrix(
                                (self.domain.n_dofs, self.domain.n_dofs), dtype=np.float64
                            )
                            for c in contact_list:
                                f_contact += c.evaluate_forces(
                                    current_nodes=self.domain.current_nodes,
                                    n_dofs=self.domain.n_dofs,
                                )
                                K_contact = K_contact + c.evaluate_stiffness(
                                    current_nodes=self.domain.current_nodes,
                                    n_dofs=self.domain.n_dofs,
                                )

                        except (ValueError, FloatingPointError):
                            self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                            augmentation.restore(multiplier_snapshot)
                            if hasattr(self.domain, "revert_state"):
                                self.domain.revert_state()
                            inner_converged = False
                            break

                        r_full = f_step + f_contact - f_int
                        r_free = r_full[free_dofs]
                        residual_norm = float(np.linalg.norm(r_free))

                        inner_tolerance = (
                            nr_tol * f_ext_free_norm if f_ref_step is not None else nr_tol
                        )
                        rel_residual = (
                            residual_norm / f_ext_free_norm if f_ref_step is not None else residual_norm
                        )

                        if verbose:
                            print(
                                f"  Iter {iteration:2d} | ||R|| = {residual_norm:.4e} | Rel = {rel_residual:.4e}"
                            )

                        if residual_norm < inner_tolerance:
                            inner_converged = True
                            if verbose:
                                print(
                                    f"  Newton equilibrium converged in {iteration} iteration(s)."
                                )
                            break

                        K_total = K_tan + K_contact
                        K_tan_free = K_total[free_dofs, :][:, free_dofs].tocsc()

                        try:
                            du_free = self.linear_solver.solve(K_tan_free, r_free)
                        except (RuntimeError, np.linalg.LinAlgError):
                            inner_converged = False
                            break
                        if not np.all(np.isfinite(du_free)):
                            inner_converged = False
                            break

                        # Line search evaluation
                        def eval_trial(alpha_cand: float) -> float:
                            u_cand = u_current.copy()
                            u_cand[free_dofs] += alpha_cand * du_free
                            if not np.all(np.isfinite(u_cand)):
                                return np.inf
                            self.domain.update_state(u=u_cand)
                            f_int_cand = self.domain.assemble_internal_forces()
                            f_c_cand = sum(
                                (
                                    c.evaluate_forces(
                                        current_nodes=self.domain.current_nodes,
                                        n_dofs=self.domain.n_dofs,
                                    )
                                    for c in contact_list
                                ),
                                np.zeros(self.domain.n_dofs, dtype=np.float64),
                            )
                            r_cand = (
                                f_step_free + f_c_cand[free_dofs] - f_int_cand[free_dofs]
                            )
                            return float(np.linalg.norm(r_cand))

                        ls_result = line_search_solver.search(
                            eval_trial,
                            residual_norm,
                            inner_tolerance,
                            contact_active=bool(contact_list),
                            reference_norm=f_ref_step,
                        )
                        if ls_result.success:
                            u_current[free_dofs] += ls_result.alpha * du_free
                        else:
                            inner_converged = False
                            break

                    total_nr_iters += iteration
                    if not inner_converged:
                        break

                    try:
                        status = augmentation.assess(self.domain.current_nodes)
                    except Exception:
                        inner_converged = False
                        break

                    if status.converged:
                        iteration_history.append(total_nr_iters)
                        al_iterations_history.append(
                            al_iteration if augmentation.contacts else 0
                        )
                        for c in contact_list:
                            if hasattr(c, "commit_state"):
                                c.commit_state()
                        if hasattr(self.domain, "commit_state"):
                            self.domain.commit_state()
                        step_converged = True
                        if verbose:
                            print(f"  [bold green]✓ Step accepted (t = {t_target:.4f} s).[/bold green]\n")
                        break

                    inner_converged = False
                    if al_iteration < augmentation.max_iterations:
                        augmentation.commit(status)

                if step_converged:
                    current_time = t_target
                    steps_completed += 1
                    time_history.append(current_time)
                    displacement_history.append(u_current.copy())

                    if callback is not None:
                        callback(steps_completed, current_time, u_current)

                    if writer is not None and (steps_completed % save_interval == 0):
                        writer.add_timestep(
                            time=current_time,
                            domain=self.domain,
                            step_index=steps_completed,
                        )

                    # Dynamic growth under adaptive stepping
                    if adaptive_stepping and total_nr_iters <= 4:
                        next_dt = min(max_dt_allowed, dt_step * gamma_grow)
                        if step_idx + 1 < len(target_dts):
                            target_dts[step_idx + 1] = next_dt

                    step_idx += 1
                else:
                    # Step failed: cutback or terminate
                    if hasattr(self.domain, "revert_state"):
                        self.domain.revert_state()
                    self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                    augmentation.restore(multiplier_snapshot)

                    if adaptive_stepping and (dt_step * gamma_cut >= dt_min):
                        dt_step *= gamma_cut
                        if verbose:
                            print(
                                f"  [bold yellow]⚠️ Newton stall. Cutting time step to dt = {dt_step:.4e} s and retrying...[/bold yellow]"
                            )
                    else:
                        if verbose:
                            print(
                                f"[bold red]✗ Step failed Newton/contact convergence at t = {current_time:.4f} s.[/bold red]"
                            )
                        all_converged = False
                        break

            if not all_converged:
                break

        # Calculate final reactions
        self.domain.update_state(u=u_current)
        f_int_final = self.domain.assemble_internal_forces()
        f_contact_final = np.zeros(self.domain.n_dofs, dtype=np.float64)
        for c in contact_list:
            f_contact_final += c.evaluate_forces(
                current_nodes=self.domain.current_nodes,
                n_dofs=self.domain.n_dofs,
            )
        reactions = f_int_final - f_contact_final - f_step

        saved_file = None
        exported_files: list[Path] = []
        if writer is not None:
            if created_internal_writer:
                writer.close()
            saved_file = getattr(writer, "output_path", None)
            exported_files = list(getattr(writer, "exported_files", []))

        t_wall_total = time.perf_counter() - t0_wall

        return QuasiStaticResult(
            displacement=u_current,
            reactions=reactions,
            converged=all_converged,
            steps_completed=steps_completed,
            time_history=time_history,
            displacement_history=displacement_history,
            iteration_history=iteration_history,
            time_series_files=exported_files,
            saved_file=saved_file,
            al_iterations_history=al_iterations_history,
            wall_time=t_wall_total,
        )
