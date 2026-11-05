from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import scipy.sparse as spsp
from rich import print

from psierra.domain.base import Domain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.boundary.contact import RigidContact, DeformableContact, SelfContact
from psierra.damping.rayleigh import RayleighDamping
from psierra.solver.base import SparseLinearSolver
from psierra.solver.sparse_linear_solver import DirectSolver
from psierra.analysis.base import Analysis
from psierra.analysis.augmentation import ContactAugmentation
from psierra.exchange import PvdCollectionWriter


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
    steady_state_converged: bool = False  # True if early equilibrium stopping occurred
    steady_state_step: int | None = None  # Step index where steady state was reached
    al_iterations_history: list[int] = field(default_factory=list)


class NonlinearDynamicAnalysis(Analysis):
    """
    Implicit Nonlinear Dynamic Solver using the Generalized-alpha method (Chung & Hulbert, 1993)
    coupled with inner Newton-Raphson iterations and backtracking line search.
    Solves M * a + C * v + F_int(u) = F_ext(t) for finite deformations with controllable numerical damping.

    The Generalized-alpha scheme unifies Newmark-beta, HHT-alpha, and Bossak-alpha.
    It is second-order accurate O(dt^2) for smooth dynamics and unconditionally
    stable for linear systems, controlled by rho_infinity in [0.0, 1.0].

    Newton trials use phi = ||R_eff||^2/2 and Armijo backtracking with
    coefficient 1e-4, at most 16 trials, and alpha halved on rejection.
    Bulk/penalty forces, inertia, and damping use the generalized midpoint.
    AL contact force instead uses (1-alpha_f)*Fc_np1+alpha_f*Fc_n, where Fc_n
    is frozen from the accepted previous endpoint. Fixed-multiplier Newton
    solves repeat until physical endpoint penetration and projected dual
    residuals pass. A passing state retains its equilibrated multipliers.
    Both the midpoint and physical
    endpoint must admit constitutive evaluation. Failure restores the last
    accepted multipliers and full (u, v, a); only accepted endpoints are exported.
    Displacements have length units, velocities length/time, accelerations
    length/time^2, and residuals force units. Sparse effective tangents combine
    the material/contact tangent with constant reduced mass and damping.
    See docs/06_solvers_and_analysis_pipelines.md; Bathe, Finite Element
    Procedures, chs. 6 and 9; Chung & Hulbert (1993).
    """

    def __init__(
        self,
        domain: Domain,
        damping: RayleighDamping | None = None,
        linear_solver: SparseLinearSolver | None = None,
        spectral_radius: float = 0.8,
    ) -> None:
        """
        Parameters:
        -----------
        domain : Domain
            Physical continuum or coupled body.
        damping : RayleighDamping, optional
            Rayleigh damping model (defaults to undamped C = 0).
        linear_solver : SparseLinearSolver, optional
            Direct or iterative sparse solver for K_eff * du = r_eff (defaults to DirectSolver).
        spectral_radius : float, default=0.8
            High-frequency spectral radius rho_infinity in [0.0, 1.0] controlling algorithmic damping.
            - rho_inf = 1.0: Undamped Newmark-beta / Midpoint Rule (alpha_m = 0.5, alpha_f = 0.5,
              gamma = 0.5, beta = 0.25). No algorithmic high-frequency damping.
              Conserves mechanical energy for undamped linear oscillations;
              nonlinear hyperelastic energy conservation is not guaranteed.
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
        self.linear_solver = (
            linear_solver if linear_solver is not None else DirectSolver()
        )
        self.spectral_radius = float(spectral_radius)

        # Generalized-alpha algorithmic constants (Chung & Hulbert, 1993)
        rho = self.spectral_radius
        self.alpha_m = (2.0 * rho - 1.0) / (rho + 1.0)
        self.alpha_f = rho / (rho + 1.0)
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        self.beta = 0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2

    def _evaluate_contacts(
        self,
        contacts: list[RigidContact | DeformableContact | SelfContact],
        old_al_force: np.ndarray,
        midpoint: tuple[np.ndarray, np.ndarray, np.ndarray],
        endpoint: tuple[np.ndarray, np.ndarray, np.ndarray],
        a1: float,
        include_stiffness: bool = True,
    ) -> tuple[np.ndarray, spsp.csr_matrix]:
        """Evaluate midpoint penalty and endpoint-weighted AL contact.

        With w=1-alpha_f, AL force is w*Fc_np1+alpha_f*Fc_n. Fc_n is frozen
        from the accepted old endpoint; the endpoint tangent gains the same w
        through the surrounding effective tangent assembly. Penalty force and
        bulk force retain midpoint geometry. Domain state is restored to the
        midpoint even when contact evaluation fails. See Popp (2012), p. 106,
        Eqs. 4.43-4.47, and docs/06_solvers_and_analysis_pipelines.md.
        """
        force = self.alpha_f * old_al_force
        tangent = spsp.csr_matrix((self.domain.n_dofs, self.domain.n_dofs))
        try:
            for augmented in (False, True):
                state = endpoint if augmented else midpoint
                self.domain.update_state(u=state[0], v=state[1], a=state[2])
                weight = 1.0 - self.alpha_f if augmented else 1.0
                for contact in contacts:
                    if (getattr(contact, "formulation", None) == "augmented_lagrangian") != augmented:
                        continue
                    force += weight * contact.evaluate_forces(
                        current_nodes=self.domain.current_nodes,
                        velocities=state[1], n_dofs=self.domain.n_dofs,
                    )
                    if include_stiffness:
                        tangent += contact.evaluate_stiffness(
                            current_nodes=self.domain.current_nodes,
                            n_dofs=self.domain.n_dofs, a1=a1,
                        )
        finally:
            self.domain.update_state(u=midpoint[0], v=midpoint[1], a=midpoint[2])
        return force, tangent

    def solve(
        self,
        dirichlet: DirichletManager | list[DirichletConstraint],
        f_ext: np.ndarray,
        dt: float,
        n_steps: int,
        contact: (
            RigidContact
            | DeformableContact
            | SelfContact
            | Sequence[RigidContact | DeformableContact | SelfContact]
            | None
        ) = None,
        nr_iterations: int = 15,
        nr_tol: float = 1e-4,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_name: str = "result",
        save_interval: int = 1,
        factor: float = 0.0,
        steady_state_tol: float | None = None,
        kinetic_energy_tol: float | None = None,
        consecutive_hold_steps: int = 3,
        verbose: bool = True,
        *,
        al_max_iterations: int = 50,
        al_gap_tol: float = 1e-6,
        al_tol: float = 1e-6,
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
            Nonnegative number of time steps. Zero exports only the initial state.
        contact : RigidContact, DeformableContact, or Sequence, optional
            Boundary contact condition(s) enforcing non-penetration against rigid obstacles or deformable bodies.
        nr_iterations : int, default=15
            Maximum Newton-Raphson iterations per time step.
        nr_tol : float, default=1e-4
            Relative dynamic residual tolerance ||R_eff|| / ||F_ext|| for Newton-Raphson equilibrium.
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
            Relative static residual tolerance ||F_ext + F_contact - F_int||
            on free DOFs, normalized by ||F_ext,f|| (replaced by 1 for a
            near-zero load). Contact is evaluated without velocity damping.
            When specified, stops the dynamic solver early once equilibrium is reached.
            If None (default), steady-state checking is ignored.
        kinetic_energy_tol : float, optional
            Relative kinetic energy decay ratio E_k / max(E_k). If provided with
            steady_state_tol, both criteria must be satisfied simultaneously.
        consecutive_hold_steps : int, default=3
            Number of consecutive steps that steady-state criteria must hold before stopping.
        verbose : bool, default=True
            Print time step progress and Newton-Raphson residual history.
        al_max_iterations : int, default=50, keyword-only
            Maximum fixed-multiplier equilibrium solves per physical time step.
        al_gap_tol : float, default=1e-6, keyword-only
            Maximum accepted physical endpoint AL penetration [length].
        al_tol : float, default=1e-6, keyword-only
            Relative infinity-norm projected multiplier tolerance. Inner force
            accuracy is tightened for both dual and gap-force targets. Small
            time steps can require more outer solves as inertial stiffness grows.

        Returns:
        --------
        result : NonlinearDynamicResult
        """
        self._validate_dynamic_controls(
            dt,
            n_steps,
            save_interval,
            steady_state_tol,
            kinetic_energy_tol,
            consecutive_hold_steps,
        )
        self._validate_count("nr_iterations", nr_iterations)
        self._validate_count("al_max_iterations", al_max_iterations)
        self._validate_scalar("al_gap_tol", al_gap_tol)
        self._validate_scalar("al_tol", al_tol)
        self._validate_scalar("nr_tol", nr_tol)
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

        if contact is None:
            if hasattr(self.domain, "contacts"):
                contact_list = list(self.domain.contacts)
            else:
                contact_list = []
        elif isinstance(contact, (RigidContact, DeformableContact, SelfContact)):
            contact_list = [contact]
        else:
            contact_list = list(contact)

        augmentation = ContactAugmentation(
            contact_list, al_max_iterations, al_gap_tol, al_tol
        )

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

        # Apply constant supports before any state-dependent force evaluation.
        # Preserve free initial displacement/velocity; supports have v_p=a_p=0.
        u_full = self.domain.u.copy()
        v_full = self.domain.v.copy()
        a_full = np.zeros(self.domain.n_dofs, dtype=np.float64)
        u_full[p_dofs] = u_p
        v_full[p_dofs] = 0.0
        self.domain.update_state(u=u_full, v=v_full, a=a_full)

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

        # Initial acceleration: M_ff * a0_f = F_ext_f + F_c0 - C_ff * v_f - F_int_0
        f_int0 = self.domain.assemble_internal_forces()
        f_c0 = sum(
            (
                c.evaluate_forces(
                    current_nodes=self.domain.current_nodes,
                    velocities=self.domain.v,
                    n_dofs=self.domain.n_dofs,
                )
                for c in contact_list
            ),
            np.zeros(self.domain.n_dofs, dtype=np.float64),
        )
        r0 = f_ext_f + f_c0[free_dofs] - C_ff @ v_f - f_int0[free_dofs]
        a_f = self.linear_solver.solve(M_ff.tocsc(), r0)

        a_full[free_dofs] = a_f
        self.domain.update_state(u=u_full, v=v_full, a=a_full)

        # 4. Prepare Time Loop Tracking
        time_points = np.linspace(0, n_steps * dt, n_steps + 1)
        exported_files: list[Path] = []
        iteration_history: list[int] = []
        al_iterations_history: list[int] = []
        pvd_writer = (
            PvdCollectionWriter(save_dir / f"{output_name}.pvd")
            if save_dir is not None
            else None
        )

        # Save Initial Step 0
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
                "[bold cyan]Nonlinear Dynamic Analysis (Implicit Generalized-alpha)[/bold cyan]"
            )
            print(
                f"Total DOFs: {self.domain.n_dofs} | Free DOFs: {len(free_dofs)} | Total Steps: {n_steps} | dt = {dt:.4e} s | rho_inf = {self.spectral_radius:.2f} | Newton Tol = {nr_tol:.1e}\n"
            )

        # 5. Dynamic Time Stepping Loop
        for step in range(1, n_steps + 1):
            t_current = step * dt
            accepted_u = self.domain.u.copy()
            accepted_v = self.domain.v.copy()
            accepted_a = self.domain.a.copy()
            multiplier_snapshot = augmentation.snapshot(self.domain.current_nodes)
            old_al_force = sum(
                (c.evaluate_forces(
                    current_nodes=self.domain.current_nodes,
                    velocities=accepted_v,
                    n_dofs=self.domain.n_dofs,
                ) for c in augmentation.contacts),
                np.zeros(self.domain.n_dofs, dtype=np.float64),
            )

            # Initial trial state for step n+1 starts at previous displacement u_n
            u_f_i = u_f.copy()
            step_converged = False
            total_iterations = 0

            for al_iteration in range(1, augmentation.max_iterations + 1):
                # Newton-Raphson equilibrium iterations within the time step
                for iteration in range(1, nr_iterations + 1):
                    # Kinematic accelerations and velocities at t_{n+1} from Newmark relations
                    a_f_i = a0 * (u_f_i - u_f) - a2 * v_f - a3 * a_f
                    v_f_i = a1 * (u_f_i - u_f) - a4 * v_f - a5 * a_f

                    # Generalized midpoint kinematics for dynamic balance
                    u_mid_f = (1.0 - alpha_f) * u_f_i + alpha_f * u_f
                    v_mid_f = (1.0 - alpha_f) * v_f_i + alpha_f * v_f
                    a_mid_f = (1.0 - alpha_m) * a_f_i + alpha_m * a_f

                    # Full vectors for internal force evaluation at midpoint
                    u_full[free_dofs] = u_mid_f
                    v_full[free_dofs] = v_mid_f
                    a_full[free_dofs] = a_mid_f
                    self.domain.update_state(u=u_full, v=v_full, a=a_full)

                    try:
                        # Assemble nonlinear internal forces F_int(u_mid)
                        f_int = self.domain.assemble_internal_forces()

                        u_endpoint = u_full.copy()
                        v_endpoint = v_full.copy()
                        a_endpoint = a_full.copy()
                        u_endpoint[free_dofs] = u_f_i
                        v_endpoint[free_dofs] = v_f_i
                        a_endpoint[free_dofs] = a_f_i
                        f_contact, K_contact = self._evaluate_contacts(
                            contact_list, old_al_force,
                            (u_full, v_full, a_full),
                            (u_endpoint, v_endpoint, a_endpoint), a1,
                        )

                    except (ValueError, FloatingPointError):
                        self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                        augmentation.restore(multiplier_snapshot)
                        if pvd_writer is not None:
                            pvd_writer.write()
                        raise

                    # Dynamic out-of-balance residual: R = F_ext + F_contact - M*a_mid - C*v_mid - F_int
                    inertia_f = M_ff @ a_mid_f
                    damping_f = C_ff @ v_mid_f
                    r_eff = (
                        f_ext_f
                        + f_contact[free_dofs]
                        - inertia_f
                        - damping_f
                        - f_int[free_dofs]
                    )

                    res_norm = float(np.linalg.norm(r_eff))
                    residual_scale = max(
                        f_ref_norm, np.linalg.norm(f_contact[free_dofs])
                    )
                    rel_res = res_norm / residual_scale
                    inner_tolerance = augmentation.inner_force_tolerance(
                        nr_tol * residual_scale, residual_scale
                    )

                    if not np.isfinite(res_norm):
                        break

                    if res_norm < inner_tolerance:
                        step_converged = True
                        break

                    # Assemble Tangent Stiffness K_tan(u_mid) and Effective Dynamic Tangent Matrix
                    try:
                        K_tan = self.domain.assemble_tangent_stiffness_matrix()
                    except (ValueError, FloatingPointError):
                        self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                        augmentation.restore(multiplier_snapshot)
                        if pvd_writer is not None:
                            pvd_writer.write()
                        raise
                    K_tan_ff = (K_tan + K_contact)[free_dofs, :][:, free_dofs]
                    K_eff = (
                        (1.0 - alpha_f) * K_tan_ff
                        + (1.0 - alpha_m) * a0 * M_ff
                        + (1.0 - alpha_f) * a1 * C_ff
                    ).tocsc()

                    # Solve for displacement increment: K_eff * du = R_eff
                    try:
                        du_f = self.linear_solver.solve(K_eff, r_eff)
                    except (RuntimeError, np.linalg.LinAlgError):
                        break
                    if not np.all(np.isfinite(du_f)):
                        break

                    # Armijo merit phi=||R_eff||^2/2, with Newton directional
                    # derivative -||R_eff||^2. Reevaluate all midpoint terms,
                    # including velocity-dependent contact; keep multipliers fixed.
                    alpha = 1.0
                    line_search_ok = False
                    for _ in range(16):
                        u_cand = u_f_i + alpha * du_f
                        a_cand = a0 * (u_cand - u_f) - a2 * v_f - a3 * a_f
                        v_cand = a1 * (u_cand - u_f) - a4 * v_f - a5 * a_f
                        u_cand_mid = (1.0 - alpha_f) * u_cand + alpha_f * u_f
                        u_full[free_dofs] = u_cand_mid
                        v_full[free_dofs] = (1.0 - alpha_f) * v_cand + alpha_f * v_f
                        a_full[free_dofs] = (1.0 - alpha_m) * a_cand + alpha_m * a_f
                        if not all(
                            np.all(np.isfinite(state))
                            for state in (u_full, v_full, a_full)
                        ):
                            alpha *= 0.5
                            continue
                        try:
                            # A valid generalized midpoint need not imply a valid
                            # physical endpoint. Guard both before accepting a trial.
                            u_endpoint = u_full.copy()
                            v_endpoint = v_full.copy()
                            a_endpoint = a_full.copy()
                            u_endpoint[free_dofs] = u_cand
                            v_endpoint[free_dofs] = v_cand
                            a_endpoint[free_dofs] = a_cand
                            self.domain.update_state(
                                u=u_endpoint, v=v_endpoint, a=a_endpoint
                            )
                            f_int_endpoint = self.domain.assemble_internal_forces()
                            if not np.all(np.isfinite(f_int_endpoint)):
                                alpha *= 0.5
                                continue
                            self.domain.update_state(u=u_full, v=v_full, a=a_full)
                            f_int_cand = self.domain.assemble_internal_forces()
                            f_contact_cand, _ = self._evaluate_contacts(
                                contact_list, old_al_force,
                                (u_full, v_full, a_full),
                                (u_endpoint, v_endpoint, a_endpoint), a1,
                                include_stiffness=False,
                            )
                            r_cand = (
                                f_ext_f + f_contact_cand[free_dofs]
                                - M_ff @ a_full[free_dofs]
                                - C_ff @ v_full[free_dofs] - f_int_cand[free_dofs]
                            )
                            cand_norm = float(np.linalg.norm(r_cand))
                        except (ValueError, FloatingPointError):
                            # Constitutive evaluation rejects an inverted trial.
                            cand_norm = np.inf
                        if np.isfinite(cand_norm) and (
                            cand_norm <= inner_tolerance
                            or cand_norm**2 <= (1.0 - 2.0e-4 * alpha) * res_norm**2
                        ):
                            u_f_i = u_cand
                            line_search_ok = True
                            break
                        else:
                            alpha *= 0.5

                    if not line_search_ok:
                        if verbose:
                            print(
                                f"  [bold red]Step {step:4d}: No admissible residual-reducing trial found.[/bold red]"
                            )
                        break

                total_iterations += iteration - 1
                if not step_converged:
                    break
                # Assess endpoint constraints; the old Newmark state and old
                # endpoint force remain frozen across every augmentation.
                a_endpoint_f = a0 * (u_f_i - u_f) - a2 * v_f - a3 * a_f
                v_endpoint_f = a1 * (u_f_i - u_f) - a4 * v_f - a5 * a_f
                u_endpoint = u_full.copy()
                v_endpoint = v_full.copy()
                a_endpoint = a_full.copy()
                u_endpoint[free_dofs] = u_f_i
                v_endpoint[free_dofs] = v_endpoint_f
                a_endpoint[free_dofs] = a_endpoint_f
                self.domain.update_state(u=u_endpoint, v=v_endpoint, a=a_endpoint)
                try:
                    status = augmentation.assess(self.domain.current_nodes)
                except (ValueError, FloatingPointError):
                    self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                    augmentation.restore(multiplier_snapshot)
                    if pvd_writer is not None:
                        pvd_writer.write()
                    raise
                if verbose and augmentation.contacts:
                    print(
                        f"  Augmentation {al_iteration}/{augmentation.max_iterations} | "
                        f"Endpoint penetration = {status.max_penetration:.4e} | "
                        f"Projected multiplier residual = {status.relative_multiplier_residual:.4e}"
                    )
                if status.converged:
                    iteration_history.append(total_iterations)
                    al_iterations_history.append(al_iteration if augmentation.contacts else 0)
                    break
                step_converged = False
                if al_iteration < augmentation.max_iterations:
                    augmentation.commit(status)

            if not step_converged:
                self.domain.update_state(u=accepted_u, v=accepted_v, a=accepted_a)
                augmentation.restore(multiplier_snapshot)
                if pvd_writer is not None:
                    pvd_writer.write()
                if verbose:
                    print(
                        f"[bold red]✗ Step {step} (t={t_current:.4f}s) failed Newton/contact "
                        f"acceptance within {al_iteration} equilibrium solve(s).[/bold red]"
                    )
                return NonlinearDynamicResult(
                    time=time_points[:step],
                    converged=False,
                    steps_completed=step - 1,
                    iteration_history=iteration_history,
                    al_iterations_history=al_iterations_history,
                    time_series_files=exported_files,
                )

            # Accept converged step: advance to physical endpoint state at t_{n+1}
            a_f_next = a0 * (u_f_i - u_f) - a2 * v_f - a3 * a_f
            v_f_next = a1 * (u_f_i - u_f) - a4 * v_f - a5 * a_f
            a_f = a_f_next
            v_f = v_f_next
            u_f = u_f_i

            u_full[free_dofs] = u_f
            v_full[free_dofs] = v_f
            a_full[free_dofs] = a_f
            self.domain.update_state(u=u_full, v=v_full, a=a_full)

            # Save frame
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
                    factor=factor,
                    include_stresses=include_stresses,
                    smooth_to_nodes=smooth_to_nodes,
                )
                pvd_writer.add_timestep(t_current, vtu_name)
                exported_files.append(vtu_path)
                saved_current_step = True

            if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps):
                max_d = np.linalg.norm(self.domain.displacement, axis=1).max()
                print(
                    f"  Step {step:5d}/{n_steps} (t = {t_current:.4f} s) | Iter = {iteration_history[-1]:2d} | Max Disp = {max_d:.3f} mm"
                )

            # Steady-state equilibrium check (dynamic relaxation / steady state)
            if steady_state_tol is not None:
                f_int_static = self.domain.assemble_internal_forces()
                f_contact_static = sum(
                    (
                        c.evaluate_forces(
                            current_nodes=self.domain.current_nodes,
                            n_dofs=self.domain.n_dofs,
                        )
                        for c in contact_list
                    ),
                    np.zeros(self.domain.n_dofs, dtype=np.float64),
                )
                r_static = f_ext_f + f_contact_static[free_dofs] - f_int_static[free_dofs]
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

        # 6. Write Master .pvd Collection
        if save_dir is not None and pvd_writer is not None:
            pvd_writer.write()

        steps_done = steady_state_step if steady_state_converged else n_steps
        time_out = time_points[: steps_done + 1]

        if verbose and not steady_state_converged:
            print(
                f"[bold green]✓ Nonlinear Dynamic Analysis completed successfully ({n_steps}/{n_steps} steps).[/bold green]\n"
            )

        return NonlinearDynamicResult(
            time=time_out,
            converged=True,
            steps_completed=steps_done,
            iteration_history=iteration_history,
            al_iterations_history=al_iterations_history,
            time_series_files=exported_files,
            steady_state_converged=steady_state_converged,
            steady_state_step=steady_state_step,
        )
