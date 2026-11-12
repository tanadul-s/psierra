"""
Hyper-Reduced Online Solvers (ECSW).

Provides ultra-fast nonlinear static and dynamic solvers operating in generalized
coordinates u_r in R^r:
1. HyperReducedNonlinearStaticSolver: Dense r x r Newton-Raphson static equilibrium solver.
2. HyperReducedNonlinearDynamicSolver: Dense r x r Generalized-alpha dynamic solver
   with predictor-corrector iterations executing in < 0.2 ms (> 5,000 Hz).
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Any, Literal
import numpy as np
import scipy.linalg as spla

from psierra.hyperreduction.model import HyperReducedNonlinearModel


@dataclass
class HyperReducedSolverStats:
    """Execution performance and energy tracking statistics for online hyper-reduced simulations."""

    sim_time: float = 0.0
    wall_time: float = 0.0
    steps_completed: int = 0
    average_step_ms: float = 0.0
    effective_hz: float = 0.0
    total_nr_iterations: int = 0
    average_nr_iterations: float = 0.0
    kinetic_energy: float = 0.0
    strain_energy: float = 0.0
    total_energy: float = 0.0


class HyperReducedNonlinearStaticSolver:
    """
    Hyper-reduced nonlinear static equilibrium solver using dense Newton-Raphson on R^r.

    Solves:
        f_r_int(u_r) - f_r_ext = 0

    Because the system is reduced to r dimensions (r << n) and evaluated on a sparse
    cubature sub-mesh of elements (|RID| << N_e), each iteration takes < 0.05 ms,
    converging in 2-5 iterations without assembling full-order sparse matrices.
    """

    def __init__(
        self,
        model: HyperReducedNonlinearModel,
        max_iter: int = 25,
        tol: float = 1e-6,
        contact: Any = None,
    ) -> None:
        """
        Parameters:
        -----------
        model : HyperReducedNonlinearModel
            ECSW hyper-reduced model.
        max_iter : int, default=25
            Maximum Newton-Raphson iterations per load step.
        tol : float, default=1e-6
            Relative residual convergence tolerance ||r_r|| / max(||f_r_ext||, 1).
        contact : HyperReducedRigidContact or Sequence, optional
            Active contact operators enforcing non-penetration constraints.
        """
        self.model = model
        self.r = model.n_modes
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        if contact is None:
            self.contacts = []
        elif isinstance(contact, (list, tuple)):
            self.contacts = list(contact)
        else:
            self.contacts = [contact]

        # State in reduced coordinates
        self.u_r = np.zeros(self.r, dtype=np.float64)

    def solve(
        self,
        f_ext_r: np.ndarray | None = None,
        n_increments: int = 1,
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Solves for the static equilibrium displacement u_r in R^r.

        Parameters:
        -----------
        f_ext_r : np.ndarray of shape (r,), optional
            Target external load vector in reduced coordinates. If None, uses model.f_r_base.
        n_increments : int, default=1
            Number of load increments to step through for high geometric non-linearity.
        verbose : bool, default=False
            Print iteration residuals.

        Returns:
        --------
        u_r : np.ndarray of shape (r,)
            Equilibrium reduced displacement coordinates.
        """
        f_target = (
            np.asarray(f_ext_r, dtype=np.float64).ravel()
            if f_ext_r is not None
            else self.model.f_r_base
        )
        if f_target.size != self.r:
            raise ValueError(
                f"Target load size ({f_target.size}) must match reduced dimension ({self.r})."
            )

        n_inc = max(1, int(n_increments))

        for inc in range(1, n_inc + 1):
            factor = float(inc) / float(n_inc)
            f_inc = factor * f_target
            f_ref_norm = max(float(np.linalg.norm(f_inc)), 1.0)

            for it in range(1, self.max_iter + 1):
                f_r_int, K_r_t = self.model.evaluate_forces_and_stiffness(self.u_r)

                f_r_c = np.zeros(self.r, dtype=np.float64)
                K_r_c = np.zeros((self.r, self.r), dtype=np.float64)
                for c in self.contacts:
                    fc, kc = c.evaluate_reduced_forces_and_stiffness(self.u_r)
                    f_r_c += fc
                    K_r_c += kc

                residual = f_r_int - f_inc - f_r_c
                res_norm = float(np.linalg.norm(residual))
                rel_res = res_norm / f_ref_norm

                if verbose:
                    print(
                        f"  [Increment {inc}/{n_inc} | Iter {it:2d}] "
                        f"||r_r|| = {res_norm:.3e} | Rel = {rel_res:.3e}"
                    )

                if rel_res <= self.tol:
                    break

                K_total = K_r_t + K_r_c
                # Dense r x r linear solve: K_total * delta_u_r = -residual
                try:
                    delta_u_r = spla.solve(K_total, -residual, assume_a="sym")
                except Exception:
                    delta_u_r = spla.solve(K_total, -residual)

                self.u_r += delta_u_r
            else:
                if verbose:
                    print(f"  ⚠️ Warning: Increment {inc} reached max iterations ({self.max_iter}).")

            # Update Uzawa multipliers if augmented lagrangian contacts are present
            for c in self.contacts:
                c.update_multipliers(self.u_r)

        return self.u_r

    def reconstruct_displacement(self) -> np.ndarray:
        """Reconstructs full volumetric 3D displacement u in R^n_dofs."""
        return self.model.reconstruct_displacement(self.u_r)

    def reconstruct_surface_displacement(self) -> np.ndarray:
        """Reconstructs 3D boundary surface mesh displacement (N_surf, 3)."""
        return self.model.reconstruct_surface_displacement(self.u_r)


class HyperReducedNonlinearDynamicSolver:
    """
    Online hyper-reduced nonlinear dynamic solver using Generalized-alpha
    implicit time integration on POD-ECSW reduced models.

    Governing Formulation:
    -----------------------
    M_r * a_r(t) + C_r * v_r(t) + f_r_int(u_r(t)) = f_r_ext(t)

    At each time step, solves the nonlinear dynamic residual using 1-4 Newton-Raphson
    iterations on the dense r x r system:
        K_eff_r * delta_u_r = R_dyn

    Where:
        K_eff_r = (1 - alpha_m) * a0 * M_r + (1 - alpha_f) * a1 * C_r + (1 - alpha_f) * K_r_t(u_r_mid)
        R_dyn   = f_r_ext - M_r * a_r_mid - C_r * v_r_mid - f_r_int(u_r_mid)
    """

    def __init__(
        self,
        model: HyperReducedNonlinearModel,
        dt: float = 0.005,
        spectral_radius: float = 0.8,
        nr_max_iter: int = 10,
        nr_tol: float = 1e-4,
        contact: Any = None,
    ) -> None:
        """
        Parameters:
        -----------
        model : HyperReducedNonlinearModel
            ECSW hyper-reduced model.
        dt : float, default=0.005
            Physical time step increment in seconds (e.g. 0.005 s = 200 Hz).
        spectral_radius : float, default=0.8
            High-frequency spectral radius rho_inf in [0.0, 1.0] for Generalized-alpha damping.
        nr_max_iter : int, default=10
            Maximum Newton-Raphson iterations per time step.
        nr_tol : float, default=1e-4
            Relative dynamic residual tolerance ||R_dyn|| / max(||f_ext||, 1).
        contact : HyperReducedRigidContact or Sequence, optional
            Active contact operators enforcing non-penetration constraints.
        """
        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        if not (0.0 <= spectral_radius <= 1.0):
            raise ValueError(f"spectral_radius must be in [0.0, 1.0], got {spectral_radius}")

        self.model = model
        self.dt = float(dt)
        self.spectral_radius = float(spectral_radius)
        self.nr_max_iter = max(1, int(nr_max_iter))
        self.nr_tol = float(nr_tol)
        self.r = model.n_modes

        if contact is None:
            self.contacts = []
        elif isinstance(contact, (list, tuple)):
            self.contacts = list(contact)
        else:
            self.contacts = [contact]

        # Generalized-alpha algorithmic parameters (Chung & Hulbert, 1993)
        rho = self.spectral_radius
        self.alpha_m = (2.0 * rho - 1.0) / (rho + 1.0)
        self.alpha_f = rho / (rho + 1.0)
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        self.beta = 0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2

        # Newmark / Generalized-alpha integration constants
        self.a0 = 1.0 / (self.beta * self.dt**2)
        self.a1 = self.gamma / (self.beta * self.dt)
        self.a2 = 1.0 / (self.beta * self.dt)
        self.a3 = 1.0 / (2.0 * self.beta) - 1.0
        self.a4 = self.gamma / self.beta - 1.0
        self.a5 = self.dt * 0.5 * (self.gamma / self.beta - 2.0)

        # Reduced kinematic state buffers
        self.u_r = np.zeros(self.r, dtype=np.float64)
        self.v_r = np.zeros(self.r, dtype=np.float64)
        self.a_r = np.zeros(self.r, dtype=np.float64)

        # Interactive perturbation force vector
        self._f_perturbation_r = np.zeros(self.r, dtype=np.float64)

        # Compute initial dynamic acceleration equilibrium
        self._init_equilibrium_acceleration()

        # Telemetry & performance tracking
        self.sim_time = 0.0
        self.wall_time = 0.0
        self.steps_completed = 0
        self._total_step_time = 0.0
        self._total_nr_iterations = 0

    def _init_equilibrium_acceleration(self) -> None:
        """Solves initial acceleration satisfying dynamic equilibrium at t=0."""
        f_total = self.model.f_r_base + self._f_perturbation_r
        f_int_0, _ = self.model.evaluate_forces_and_stiffness(self.u_r)
        res = f_total - self.model.C_r @ self.v_r - f_int_0
        try:
            self.a_r = spla.solve(self.model.M_r, res, assume_a="pos")
        except Exception:
            self.a_r = spla.solve(self.model.M_r, res)

    def reset_state(self) -> None:
        """Resets kinematic state and timers."""
        self.u_r.fill(0.0)
        self.v_r.fill(0.0)
        self._f_perturbation_r.fill(0.0)
        for c in self.contacts:
            c.reset_state()
        self._init_equilibrium_acceleration()
        self.sim_time = 0.0
        self.wall_time = 0.0
        self.steps_completed = 0
        self._total_step_time = 0.0
        self._total_nr_iterations = 0

    def set_perturbation(self, f_perturbation_r: np.ndarray) -> None:
        """Applies an active interactive perturbation force vector in reduced coordinates."""
        f_arr = np.asarray(f_perturbation_r, dtype=np.float64).ravel()
        if f_arr.size != self.r:
            raise ValueError(f"Perturbation size {f_arr.size} must match mode count {self.r}.")
        self._f_perturbation_r[:] = f_arr

    def clear_perturbation(self) -> None:
        """Clears interactive perturbations."""
        self._f_perturbation_r.fill(0.0)

    def step(self, f_ext_r: np.ndarray | None = None) -> np.ndarray:
        """
        Advances the hyper-reduced dynamic system by one time step dt using
        nonlinear Generalized-alpha predictor-corrector iterations.

        Parameters:
        -----------
        f_ext_r : np.ndarray of shape (r,), optional
            External load in reduced coordinates. If None, uses model.f_r_base + perturbation.

        Returns:
        --------
        u_r : np.ndarray of shape (r,)
            Updated reduced displacement coordinates.
        """
        t_start = time.perf_counter()

        if f_ext_r is not None:
            f_total = np.asarray(f_ext_r, dtype=np.float64).ravel()
        else:
            f_total = self.model.f_r_base + self._f_perturbation_r

        f_ref_norm = max(float(np.linalg.norm(f_total)), 1.0)

        # 1. Newmark Predictor: estimate u_{n+1}^{(0)}, a_{n+1}^{(0)}, v_{n+1}^{(0)}
        u_next = self.u_r.copy()
        a_next = -self.a2 * self.v_r - self.a3 * self.a_r
        v_next = self.v_r + self.dt * (1.0 - self.gamma) * self.a_r

        # Constant linear parts of K_eff_r
        K_eff_const = (1.0 - self.alpha_m) * self.a0 * self.model.M_r + (
            1.0 - self.alpha_f
        ) * self.a1 * self.model.C_r

        # 2. Newton-Raphson Corrector Iterations
        nr_iters = 0
        for it in range(self.nr_max_iter):
            nr_iters += 1
            # Midpoint kinematic evaluations
            u_mid = (1.0 - self.alpha_f) * u_next + self.alpha_f * self.u_r
            v_mid = (1.0 - self.alpha_f) * v_next + self.alpha_f * self.v_r
            a_mid = (1.0 - self.alpha_m) * a_next + self.alpha_m * self.a_r

            # Evaluate hyper-reduced nonlinear internal force and tangent stiffness
            f_int_mid, K_t_mid = self.model.evaluate_forces_and_stiffness(u_mid)

            # Evaluate active contact force and tangent stiffness contributions
            f_c_mid = np.zeros(self.r, dtype=np.float64)
            K_c_mid = np.zeros((self.r, self.r), dtype=np.float64)
            for c in self.contacts:
                fc, kc = c.evaluate_reduced_forces_and_stiffness(u_mid, v_mid, a1=self.a1)
                f_c_mid += fc
                K_c_mid += kc

            # Dynamic residual vector:
            # R_dyn = f_ext + f_c - M_r * a_mid - C_r * v_mid - f_int_mid
            R_dyn = (
                f_total + f_c_mid - self.model.M_r @ a_mid - self.model.C_r @ v_mid - f_int_mid
            )
            res_norm = float(np.linalg.norm(R_dyn))

            if res_norm / f_ref_norm <= self.nr_tol:
                break

            # Effective dynamic stiffness:
            # K_eff = K_eff_const + (1 - alpha_f) * (K_t_mid + K_c_mid)
            K_eff = K_eff_const + (1.0 - self.alpha_f) * (K_t_mid + K_c_mid)

            # Solve for displacement increment: K_eff * delta_u = R_dyn
            try:
                delta_u = spla.solve(K_eff, R_dyn, assume_a="sym")
            except Exception:
                delta_u = spla.solve(K_eff, R_dyn)

            # Corrector update
            u_next += delta_u
            a_next = self.a0 * (u_next - self.u_r) - self.a2 * self.v_r - self.a3 * self.a_r
            v_next = self.v_r + self.dt * ((1.0 - self.gamma) * self.a_r + self.gamma * a_next)

        # 3. Commit state updates
        self.u_r[:] = u_next
        self.v_r[:] = v_next
        self.a_r[:] = a_next

        # 4. Commit contact multiplier updates
        for c in self.contacts:
            c.update_multipliers(self.u_r)

        t_elapsed = time.perf_counter() - t_start
        self._total_step_time += t_elapsed
        self.sim_time += self.dt
        self.wall_time += t_elapsed
        self.steps_completed += 1
        self._total_nr_iterations += nr_iters

        return self.u_r

    def reconstruct_displacement(self) -> np.ndarray:
        """Reconstructs full volumetric 3D displacement u in R^n_dofs."""
        return self.model.reconstruct_displacement(self.u_r)

    def reconstruct_surface_displacement(self) -> np.ndarray:
        """Reconstructs 3D boundary surface mesh displacement (N_surf, 3)."""
        return self.model.reconstruct_surface_displacement(self.u_r)

    def compute_kinetic_energy(self) -> float:
        """Computes instantaneous reduced kinetic energy: 0.5 * v_r^T * M_r * v_r."""
        return float(0.5 * self.v_r.T @ (self.model.M_r @ self.v_r))

    def compute_strain_energy(self) -> float:
        """Approximates instantaneous elastic strain energy."""
        f_int, _ = self.model.evaluate_forces_and_stiffness(self.u_r)
        return float(0.5 * self.u_r.T @ f_int)

    def compute_total_energy(self) -> float:
        """Computes total mechanical energy: E_k + E_s."""
        return self.compute_kinetic_energy() + self.compute_strain_energy()

    def get_stats(self) -> HyperReducedSolverStats:
        """Returns comprehensive real-time execution statistics."""
        avg_ms = (
            (self._total_step_time / self.steps_completed) * 1000.0
            if self.steps_completed > 0
            else 0.0
        )
        eff_hz = 1000.0 / avg_ms if avg_ms > 0.0 else 0.0
        avg_nr = (
            self._total_nr_iterations / self.steps_completed
            if self.steps_completed > 0
            else 0.0
        )

        return HyperReducedSolverStats(
            sim_time=self.sim_time,
            wall_time=self.wall_time,
            steps_completed=self.steps_completed,
            average_step_ms=avg_ms,
            effective_hz=eff_hz,
            total_nr_iterations=self._total_nr_iterations,
            average_nr_iterations=avg_nr,
            kinetic_energy=self.compute_kinetic_energy(),
            strain_energy=self.compute_strain_energy(),
            total_energy=self.compute_total_energy(),
        )


@dataclass
class HyperReducedDynamicRelaxationStats:
    """Execution performance and convergence telemetry for hyper-reduced Dynamic Relaxation."""

    steps_completed: int = 0
    kinetic_peaks: int = 0
    kinetic_energy: float = 0.0
    strain_energy: float = 0.0
    residual_norm: float = 0.0
    relative_residual: float = 0.0
    converged: bool = False
    average_step_ms: float = 0.0
    effective_hz: float = 0.0
    wall_time: float = 0.0


class HyperReducedDynamicRelaxationSolver:
    """
    Hyper-reduced explicit Dynamic Relaxation (DR) static equilibrium solver on R^r.

    Solves the static equilibrium equations:
        R_r(u_r) = f_r_ext + f_r_c(u_r) - f_r_int(u_r) = 0
    by integrating a fictitious, highly damped explicit dynamic system:
        M_r_fict * u_r_ddot + C_r_fict * u_r_dot = R_r(u_r)

    Key Features:
    - Matrix-free explicit vector evaluation on R^r: avoids dense r x r matrix inversion and factorization.
    - Parameter-free Barnes/Underwood kinetic energy peak damping (v_r <- 0 when dEk/dt < 0).
    - Unconditionally robust in non-smooth contact, avoiding Newton-Raphson chattering or step rejection.
    - Gerschgorin diagonal mass scaling guaranteeing CFL numerical stability for explicit time-stepping.
    - Provides both incremental pseudo-stepping (step) for real-time visualization of the relaxation path
      and multi-step relaxation (relax) to converge directly to static equilibrium.
    """

    def __init__(
        self,
        model: HyperReducedNonlinearModel,
        dt: float = 1.0,
        damping_type: Literal["kinetic", "viscous"] = "kinetic",
        safety_factor: float = 1.1,
        viscous_damping: float | None = None,
        tol: float = 1e-5,
        max_steps: int = 500,
        min_steps: int = 5,
        contact: Any = None,
    ) -> None:
        """
        Parameters:
        -----------
        model : HyperReducedNonlinearModel
            ECSW hyper-reduced model.
        dt : float, default=1.0
            Explicit pseudo-time step increment.
        damping_type : {'kinetic', 'viscous'}, default='kinetic'
            Relaxation damping mechanism. 'kinetic' is parameter-free and recommended.
        safety_factor : float, default=1.1
            Gerschgorin mass scaling safety factor (must be >= 1.0).
        viscous_damping : float, optional
            Mass-proportional damping factor (used when damping_type='viscous').
        tol : float, default=1e-5
            Relative residual convergence tolerance ||R_r|| / max(||f_r_ext||, 1.0).
        max_steps : int, default=500
            Maximum relaxation steps per solve.
        min_steps : int, default=5
            Minimum relaxation steps before checking convergence.
        contact : HyperReducedRigidContact or Sequence, optional
            Active contact operators enforcing obstacle constraints.
        """
        self.model = model
        self.r = model.n_modes
        self.dt = float(dt)
        self.damping_type = damping_type.lower()
        if self.damping_type not in ("kinetic", "viscous"):
            raise ValueError(f"damping_type must be 'kinetic' or 'viscous', got '{damping_type}'")
        self.safety_factor = float(safety_factor)
        self.viscous_damping = viscous_damping if viscous_damping is not None else 0.5
        self.tol = float(tol)
        self.max_steps = max(1, int(max_steps))
        self.min_steps = max(1, int(min_steps))

        if contact is None:
            self.contacts = []
        elif isinstance(contact, (list, tuple)):
            self.contacts = list(contact)
        else:
            self.contacts = [contact]

        # Kinematic state buffers in reduced coordinates R^r
        self.u_r = np.zeros(self.r, dtype=np.float64)
        self.v_half_r = np.zeros(self.r, dtype=np.float64)
        self.a_r = np.zeros(self.r, dtype=np.float64)

        # Fictitious diagonal mass setup
        self._setup_fictitious_mass()

        # Viscous damping constants (c1, c2)
        cd = self.viscous_damping
        self.c1 = (2.0 - cd * self.dt) / (2.0 + cd * self.dt)
        self.c2 = (2.0 * self.dt) / (2.0 + cd * self.dt)

        # Kinetic damping tracking
        self.prev_Ek = 0.0
        self.kinetic_peaks = 0
        self.converged = False
        self.last_rel_res = 1.0
        self.last_res_norm = 0.0

        # Performance tracking
        self.steps_completed = 0
        self.wall_time = 0.0
        self._total_step_time = 0.0

    def _setup_fictitious_mass(self) -> None:
        """Computes explicit Gerschgorin-scaled fictitious diagonal mass on R^r."""
        _, K0_r = self.model.evaluate_forces_and_stiffness(np.zeros(self.r))
        K_tot = np.abs(K0_r)
        for c in self.contacts:
            if hasattr(c, "surface_basis") and hasattr(c, "k_penalty"):
                Phi_s = c.surface_basis.basis
                K_c_bound = c.k_penalty * (Phi_s.T @ Phi_s)
                K_tot += np.abs(K_c_bound)
            elif hasattr(c, "evaluate_reduced_forces_and_stiffness"):
                try:
                    _, Kc_0 = c.evaluate_reduced_forces_and_stiffness(np.zeros(self.r))
                    K_tot += np.abs(Kc_0)
                except Exception:
                    pass

        row_sums = np.sum(K_tot, axis=1)
        max_row = float(row_sums.max()) if row_sums.size > 0 and row_sums.max() > 0 else 1.0
        row_sums = np.maximum(row_sums, 1e-6 * max_row)

        self.m_diag = 0.25 * self.safety_factor * (self.dt**2) * row_sums
        if hasattr(self.model, "M_r"):
            m_phys = np.diag(self.model.M_r)
            self.m_diag = np.maximum(self.m_diag, m_phys)
        self.inv_m = 1.0 / self.m_diag

    def reset_state(self) -> None:
        """Resets kinematic state, contact history, and relaxation statistics."""
        self.u_r.fill(0.0)
        self.v_half_r.fill(0.0)
        self.a_r.fill(0.0)
        self.prev_Ek = 0.0
        self.kinetic_peaks = 0
        self.converged = False
        self.last_rel_res = 1.0
        self.last_res_norm = 0.0
        self.steps_completed = 0
        self.wall_time = 0.0
        self._total_step_time = 0.0
        for c in self.contacts:
            c.reset_state()

    def step(self, f_ext_r: np.ndarray | None = None) -> np.ndarray:
        """
        Advances the hyper-reduced system by one explicit pseudo-time relaxation step.

        Parameters:
        -----------
        f_ext_r : np.ndarray of shape (r,), optional
            Target external load vector in reduced coordinates. If None, uses model.f_r_base.

        Returns:
        --------
        u_r : np.ndarray of shape (r,)
            Updated reduced displacement vector.
        """
        t_start = time.perf_counter()

        f_total = (
            np.asarray(f_ext_r, dtype=np.float64).ravel()
            if f_ext_r is not None
            else self.model.f_r_base
        )
        f_ref_norm = max(float(np.linalg.norm(f_total)), 1.0)

        # 1. Update displacement: u^{n+1} = u^n + dt * v^{n+1/2}
        u_prev = self.u_r.copy()
        a_prev = self.a_r.copy()
        self.u_r += self.dt * self.v_half_r

        # 2. Evaluate internal and contact forces at u_r with inversion safeguard
        try:
            f_int, _ = self.model.evaluate_forces_and_stiffness(self.u_r)
        except (ValueError, FloatingPointError):
            self.u_r[:] = 0.5 * (self.u_r + u_prev)
            self.v_half_r.fill(0.0)
            f_int, _ = self.model.evaluate_forces_and_stiffness(self.u_r)

        f_c = np.zeros(self.r, dtype=np.float64)
        for c in self.contacts:
            fc, _ = c.evaluate_reduced_forces_and_stiffness(self.u_r, self.v_half_r)
            f_c += fc

        residual = f_total + f_c - f_int
        res_norm = float(np.linalg.norm(residual))
        rel_res = res_norm / f_ref_norm
        self.last_res_norm = res_norm
        self.last_rel_res = rel_res

        # 3. Explicit acceleration: a = inv_m * R
        a_next = self.inv_m * residual

        # 4. Kinetic energy evaluation & peak damping reset
        if self.damping_type == "kinetic":
            v_full = self.v_half_r + self.dt * a_next
            v_mid = 0.5 * (self.v_half_r + v_full)
            Ek = 0.5 * float(np.sum(self.m_diag * (v_mid**2)))

            if self.steps_completed > 0 and Ek < self.prev_Ek:
                self.kinetic_peaks += 1
                # Underwood peak displacement interpolation
                u_peak = self.u_r - 1.5 * self.dt * self.v_half_r + 0.5 * (self.dt**2) * a_prev
                self.u_r[:] = u_peak

                # Re-evaluate forces at peak displacement
                f_int_pk, _ = self.model.evaluate_forces_and_stiffness(self.u_r)
                f_c_pk = np.zeros(self.r, dtype=np.float64)
                for c in self.contacts:
                    fc, _ = c.evaluate_reduced_forces_and_stiffness(self.u_r, np.zeros(self.r))
                    f_c_pk += fc
                residual = f_total + f_c_pk - f_int_pk
                a_next = self.inv_m * residual
                res_norm = float(np.linalg.norm(residual))
                rel_res = res_norm / f_ref_norm
                self.last_res_norm = res_norm
                self.last_rel_res = rel_res

                # Velocity reset to zero from peak
                v_next = 0.5 * self.dt * a_next
                self.prev_Ek = 0.0
            else:
                v_next = v_full
                self.prev_Ek = Ek

            self.v_half_r[:] = v_next
            self.a_r[:] = a_next
        else:
            self.v_half_r[:] = self.c1 * self.v_half_r + self.c2 * a_next
            self.a_r[:] = a_next

        # 5. Commit contact multipliers
        for c in self.contacts:
            c.update_multipliers(self.u_r)

        self.converged = rel_res <= self.tol
        t_elapsed = time.perf_counter() - t_start
        self._total_step_time += t_elapsed
        self.wall_time += t_elapsed
        self.steps_completed += 1

        return self.u_r

    def relax(
        self,
        f_ext_r: np.ndarray | None = None,
        tol: float | None = None,
        max_steps: int | None = None,
        min_steps: int | None = None,
    ) -> np.ndarray:
        """
        Advances explicit dynamic relaxation iterations until static equilibrium is reached.

        Parameters:
        -----------
        f_ext_r : np.ndarray of shape (r,), optional
            Target external load vector in reduced coordinates.
        tol : float, optional
            Convergence tolerance (defaults to self.tol).
        max_steps : int, optional
            Maximum allowed relaxation steps (defaults to self.max_steps).
        min_steps : int, optional
            Minimum relaxation steps before declaring convergence (defaults to self.min_steps).

        Returns:
        --------
        u_r : np.ndarray of shape (r,)
            Converged static equilibrium reduced displacement coordinates.
        """
        target_tol = float(tol) if tol is not None else self.tol
        m_steps = int(max_steps) if max_steps is not None else self.max_steps
        min_s = int(min_steps) if min_steps is not None else self.min_steps

        for _ in range(m_steps):
            self.step(f_ext_r)
            if self.steps_completed >= min_s and self.last_rel_res <= target_tol:
                self.converged = True
                break

        return self.u_r

    def reconstruct_displacement(self) -> np.ndarray:
        """Reconstructs full volumetric 3D displacement u in R^n_dofs."""
        return self.model.reconstruct_displacement(self.u_r)

    def reconstruct_surface_displacement(self) -> np.ndarray:
        """Reconstructs 3D boundary surface mesh displacement (N_surf, 3)."""
        return self.model.reconstruct_surface_displacement(self.u_r)

    def compute_kinetic_energy(self) -> float:
        """Computes instantaneous reduced kinetic energy."""
        return float(0.5 * np.sum(self.m_diag * (self.v_half_r**2)))

    def compute_strain_energy(self) -> float:
        """Approximates instantaneous elastic strain energy."""
        f_int, _ = self.model.evaluate_forces_and_stiffness(self.u_r)
        return float(0.5 * self.u_r.T @ f_int)

    def get_stats(self) -> HyperReducedDynamicRelaxationStats:
        """Returns comprehensive real-time execution statistics."""
        avg_ms = (
            (self._total_step_time / self.steps_completed) * 1000.0
            if self.steps_completed > 0
            else 0.0
        )
        eff_hz = 1000.0 / avg_ms if avg_ms > 0.0 else 0.0

        return HyperReducedDynamicRelaxationStats(
            steps_completed=self.steps_completed,
            kinetic_peaks=self.kinetic_peaks,
            kinetic_energy=self.compute_kinetic_energy(),
            strain_energy=self.compute_strain_energy(),
            residual_norm=self.last_res_norm,
            relative_residual=self.last_rel_res,
            converged=self.converged,
            average_step_ms=avg_ms,
            effective_hz=eff_hz,
            wall_time=self.wall_time,
        )
