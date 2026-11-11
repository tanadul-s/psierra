from __future__ import annotations
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
    saved_file: Path | None = None  # Saved primary visualization file (e.g. result.xdmf)
    al_iterations_history: list[int] = field(default_factory=list)


class NonlinearStaticAnalysis(Analysis):
    """
    Nonlinear Static Solver using Incremental Newton-Raphson with Backtracking Line Search.
    Solves R(U) = lambda * F_ext - F_int(U) = 0 using Total Lagrangian kinematics.

    With contact, R = lambda F_ext + F_contact - F_int (force units).
    The sparse tangent solves for displacement increments (length units).
    Armijo backtracking uses phi = ||R_f||^2/2, coefficient 1e-4 and at most
    16 trials with alpha halved on rejection. Contact forces are reevaluated
    while multipliers remain fixed. AL outer iterations repeat equilibrium
    after projected multiplier updates until penetration and relative projected
    multiplier defects pass their tolerances. A passing state retains its solved
    multipliers without another update. Failure restores accepted multipliers
    and full (u, v, a); reactions use that accepted load factor. Supplied initial
    free displacements are retained; fixed constraints ramp with the load,
    while callable constraints supply their own step values.
    See docs/06_solvers_and_analysis_pipelines.md and Bathe, Finite Element
    Procedures, ch. 6 (nonlinear solution procedures).
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
        contact: (
            ContactMultiplierState
            | Sequence[ContactMultiplierState]
            | None
        ) = None,
        n_steps: int = 10,
        nr_iterations: int = 25,
        nr_tol: float = 1e-4,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_name: str = "result",
        format: str = "xdmf",
        save_interval: int = 1,
        save_history: bool = True,
        factor: float = 0.0,
        verbose: bool = True,
        writer: SimulationWriter | None = None,
        initial_step: int = 0,
        initial_time: float = 0.0,
        *,
        al_max_iterations: int = 50,
        al_gap_tol: float = 1e-6,
        al_tol: float = 1e-6,
        rebound_relaxation: bool = False,
        line_search: ContactAwareLineSearch | None = None,
    ) -> NonlinearStaticResult:
        """
        Solves the nonlinear equilibrium equation incrementally.

        Parameters:
        -----------
        dirichlet : DirichletManager, list of DirichletBoundaryCondition, or callable (step, n_steps) -> DirichletManager
            Essential boundary conditions. Can be a static constraint manager or a step-dependent callable.
        f_ext : np.ndarray of shape (n_dofs,), optional
            Total external load vector to be reached at lambda = 1.0. Defaults to zero vector for displacement control.
        contact : RigidContact, DeformableContact, or Sequence, optional
            Boundary contact condition(s) enforcing non-penetration against rigid obstacles or deformable bodies.
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
            Directory to save visualization files to.
        output_name : str, default="result"
            Master filename prefix for exported result files (result.xdmf or result.pvd).
        save_interval : int, optional
            Number of steps between saved intermediate load frames when history is requested.
        save_history : bool, default=False
            If True, records the full multi-step load ramp history.
            If False (default), records ONLY the final converged static equilibrium state
            (matching LinearStaticAnalysis and DynamicRelaxationAnalysis in ParaView).
        verbose : bool, default=True
            Print step-by-step residual and convergence reports.
        al_max_iterations : int, default=50, keyword-only
            Maximum fixed-multiplier equilibrium solves per load increment.
        al_gap_tol : float, default=1e-6, keyword-only
            Maximum accepted AL penetration [length].
        al_tol : float, default=1e-6, keyword-only
            Relative infinity-norm projected multiplier tolerance. Inner force
            accuracy is tightened to resolve both dual and gap-force targets.

        Returns:
        --------
        result : NonlinearStaticResult
        """
        self._validate_count("n_steps", n_steps)
        self._validate_count("nr_iterations", nr_iterations)
        self._validate_count("al_max_iterations", al_max_iterations)
        self._validate_scalar("al_gap_tol", al_gap_tol)
        self._validate_scalar("al_tol", al_tol)
        self._validate_count("save_interval", save_interval)
        record_history = bool(save_history)
        stride = save_interval
        self._validate_scalar("nr_tol", nr_tol)
        if f_ext is None:
            f_ext = np.zeros(self.domain.n_dofs, dtype=np.float64)
        f_ext = np.asarray(f_ext, dtype=np.float64)
        if f_ext.shape != (self.domain.n_dofs,) or not np.all(np.isfinite(f_ext)):
            raise ValueError("f_ext must be a finite vector of shape (n_dofs,)")

        if contact is None:
            if hasattr(self.domain, "contacts"):
                contact_list = list(self.domain.contacts)
            else:
                contact_list = []
        elif isinstance(contact, ContactMultiplierState):
            contact_list = [contact]
        else:
            contact_list = list(contact)

        augmentation = ContactAugmentation(
            contact_list,
            al_max_iterations,
            al_gap_tol,
            al_tol,
            rebound_relaxation=rebound_relaxation,
        )
        line_search_solver = (
            line_search if line_search is not None else ContactAwareLineSearch()
        )

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
        created_internal_writer = False
        if record_history:
            if save_dir is not None:
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

            # Save Step 0 (Initial state at lambda = 0.0) only when recording history
            if writer is not None and initial_step == 0:
                writer.add_timestep(time=initial_time, domain=self.domain, step_index=0)

        u_current = self.domain.u.copy()
        iteration_history: list[int] = []
        al_iterations_history: list[int] = []
        exported_files: list[Path] = []

        if verbose:
            print(
                "[bold cyan]Nonlinear Static Analysis (Incremental Newton-Raphson)[/bold cyan]"
            )
            print(
                f"Total DOFs: {len(u_current)} | Free DOFs: {len(free_dofs)} | Load Steps: {n_steps} | Max Iter/Step: {nr_iterations} | Tol: {nr_tol:.1e}\n"
            )

        for step in range(1, n_steps + 1):
            load_factor = step / n_steps
            accepted_u = self.domain.u.copy()
            accepted_v = self.domain.v.copy()
            accepted_a = self.domain.a.copy()
            if augmentation.rebound_relaxation:
                augmentation.relax_rebound()
            multiplier_snapshot = augmentation.snapshot(self.domain.current_nodes)

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
            elif len(p_dofs) > 0:
                u_current[p_dofs] = load_factor * u_p

            f_step = load_factor * f_ext
            f_step_free = f_step[free_dofs]
            f_ext_free_norm = float(np.linalg.norm(f_step_free))
            f_ref_step: float | None = (
                f_ext_free_norm if f_ext_free_norm >= 1e-12 else None
            )

            if verbose:
                print(
                    f"[bold cyan]--- Step {step:2d}/{n_steps} (Load Factor lambda = {load_factor:.2f}) ---[/bold cyan]"
                )

            converged_step = False
            total_iterations = 0

            for al_iteration in range(1, augmentation.max_iterations + 1):
                for iteration in range(1, nr_iterations + 1):
                    # 1. Update state in domain with current displacement guess
                    self.domain.update_state(u=u_current)

                    try:
                        # 2. Assemble internal forces F_int(U) and tangent stiffness K_tan(U)
                        f_int = self.domain.assemble_internal_forces()
                        K_tan = self.domain.assemble_tangent_stiffness_matrix()

                        # Contact force and tangent stiffness
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
                        if writer is not None:
                            writer.close()
                        raise

                    # 3. Compute Out-of-Balance Residual Vector: R = F_ext_step + F_contact - F_int
                    r_free = f_step_free + f_contact[free_dofs] - f_int[free_dofs]
                    residual_norm = float(np.linalg.norm(r_free))
                    if not np.isfinite(residual_norm):
                        break

                    if f_ref_step is None:
                        # Displacement-controlled problem without external loads:
                        # normalize against the step's initial unbalance or internal force scale
                        f_ref_step = max(
                            residual_norm, float(np.linalg.norm(f_int[free_dofs])), 1.0
                        )

                    rel_residual = residual_norm / f_ref_step
                    inner_tolerance = augmentation.inner_force_tolerance(
                        nr_tol * f_ref_step, f_ref_step
                    )

                    if verbose:
                        print(
                            f"  Iter {iteration:2d} | ||R|| = {residual_norm:.4e} | Rel = {rel_residual:.4e}"
                        )

                    # Check Convergence
                    if residual_norm < inner_tolerance:
                        converged_step = True
                        if verbose:
                            print(
                                    f"  Newton equilibrium converged in {iteration} iteration(s)."
                            )
                        break

                    # 4. Partition Tangent Stiffness to Free DOFs
                    K_total = K_tan + K_contact
                    K_tan_free = K_total[free_dofs, :][:, free_dofs].tocsc()

                    # 5. Solve for Displacement Increment: K_tan_free * du = r_free
                    try:
                        du_free = self.linear_solver.solve(K_tan_free, r_free)
                    except (RuntimeError, np.linalg.LinAlgError):
                        break
                    if not np.all(np.isfinite(du_free)):
                        break

                    # 6. Contact-aware line search with robust FEA backtracking.
                    def eval_static_trial(alpha_cand: float) -> float:
                        u_cand = u_current.copy()
                        u_cand[free_dofs] += alpha_cand * du_free
                        if not np.all(np.isfinite(u_cand)):
                            return np.inf
                        self.domain.update_state(u=u_cand)
                        f_int_candidate = self.domain.assemble_internal_forces()
                        f_contact_candidate = sum(
                            (
                                c.evaluate_forces(
                                    current_nodes=self.domain.current_nodes,
                                    n_dofs=self.domain.n_dofs,
                                )
                                for c in contact_list
                            ),
                            np.zeros(self.domain.n_dofs, dtype=np.float64),
                        )
                        r_candidate = (
                            f_step_free + f_contact_candidate[free_dofs]
                            - f_int_candidate[free_dofs]
                        )
                        return float(np.linalg.norm(r_candidate))

                    ls_result = line_search_solver.search(
                        eval_static_trial,
                        residual_norm,
                        inner_tolerance,
                        contact_active=bool(contact_list),
                    )
                    if ls_result.success:
                        u_current[free_dofs] += ls_result.alpha * du_free
                    else:
                        if verbose:
                            print(
                                f"  [bold red]Step {step:2d}/{n_steps}: Line search failed ({ls_result.rejection_reason}).[/bold red]"
                            )
                        converged_step = False
                        break

                total_iterations += iteration
                if not converged_step:
                    break
                try:
                    status = augmentation.assess(self.domain.current_nodes)
                except (ValueError, FloatingPointError):
                    self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                    augmentation.restore(multiplier_snapshot)
                    if writer is not None:
                        writer.close()
                    raise
                if verbose and augmentation.contacts:
                    print(
                        f"  Augmentation {al_iteration}/{augmentation.max_iterations} | "
                        f"Penetration = {status.max_penetration:.4e} | "
                        f"Projected multiplier residual = {status.relative_multiplier_residual:.4e}"
                    )
                if status.converged:
                    iteration_history.append(total_iterations)
                    al_iterations_history.append(al_iteration if augmentation.contacts else 0)
                    if verbose:
                        print(f"  [bold green]✓ Step {step} accepted.[/bold green]\n")
                    break
                converged_step = False
                if al_iteration < augmentation.max_iterations:
                    augmentation.commit(status)

            if not converged_step:
                if verbose:
                    print(
                        f"[bold red]✗ Step {step} failed Newton/contact acceptance "
                        f"within {al_iteration} equilibrium solve(s).[/bold red]"
                    )
                u_current = accepted_u
                self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                augmentation.restore(multiplier_snapshot)
                if created_internal_writer and writer is not None:
                    writer.close()
                elif writer is not None and hasattr(writer, "write"):
                    writer.write()
                f_contact_step = sum(
                    (
                        c.evaluate_forces(
                            current_nodes=self.domain.current_nodes,
                            n_dofs=self.domain.n_dofs,
                        )
                        for c in contact_list
                    ),
                    np.zeros(self.domain.n_dofs, dtype=np.float64),
                )
                reactions = (
                    self.domain.assemble_internal_forces()
                    - ((step - 1) / n_steps) * f_ext
                    - f_contact_step
                )
                return NonlinearStaticResult(
                    displacement=u_current,
                    reactions=reactions,
                    converged=False,
                    steps_completed=step - 1,
                    iteration_history=iteration_history,
                    al_iterations_history=al_iterations_history,
                    time_series_files=getattr(writer, "exported_files", []) if writer is not None else [],
                )

            # Save frame if history recording requested and matching stride or final step
            step_idx = initial_step + step
            t_current = initial_time + float(load_factor)
            if record_history and writer is not None and (step % stride == 0 or step == n_steps):
                writer.add_timestep(time=t_current, domain=self.domain, step_index=step_idx)

        # Simulation successfully completed all steps
        self.domain.update_state(u=u_current)
        f_int_final = self.domain.assemble_internal_forces()
        f_contact_final = sum(
            (
                c.evaluate_forces(
                    current_nodes=self.domain.current_nodes, n_dofs=self.domain.n_dofs
                )
                for c in contact_list
            ),
            np.zeros(self.domain.n_dofs, dtype=np.float64),
        )
        reactions = f_int_final - f_ext - f_contact_final

        # Finalize writer output
        if not record_history:
            # Default static solve: save converged static equilibrium state directly (matching LinearStaticAnalysis)
            if save_dir is not None and writer is None:
                save_dir.mkdir(parents=True, exist_ok=True)
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
                writer.add_timestep(time=0.0, domain=self.domain, step_index=0)
                if created_internal_writer:
                    writer.close()
                elif hasattr(writer, "write"):
                    writer.write()
                exported_files = getattr(writer, "exported_files", [])
            else:
                exported_files = []
        else:
            if writer is not None:
                if created_internal_writer:
                    writer.close()
                elif hasattr(writer, "write"):
                    writer.write()
                exported_files = getattr(writer, "exported_files", [])
            else:
                exported_files = []

        saved_file = exported_files[0] if exported_files else None

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
            al_iterations_history=al_iterations_history,
            time_series_files=exported_files,
            saved_file=saved_file,
        )
