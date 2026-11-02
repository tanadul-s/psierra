from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, field
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spspla

from psierra.analysis.base import Analysis
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint, DirichletBoundaryCondition
from psierra.damping.rayleigh import RayleighDamping
from psierra.domain.base import Domain


@dataclass
class SimulationStats:
    """Execution performance and energy tracking statistics for online dynamic simulation."""

    sim_time: float = 0.0
    wall_time: float = 0.0
    steps_completed: int = 0
    average_step_ms: float = 0.0
    speed_ratio: float = 0.0
    effective_hz: float = 0.0
    kinetic_energy: float = 0.0
    strain_energy: float = 0.0
    total_energy: float = 0.0


class InteractiveDynamicSimulation(Analysis):
    """
    High-performance real-time dynamic simulation pipeline using Generalized-alpha
    implicit time integration with single-factorization SuperLU reuse.

    Features:
    ---------
    - Pre-factorizes effective dynamic stiffness K_eff once at setup for ultra-fast O(N) back-substitutions (>10,000 Hz physics rates).
    - Unconditionally stable Generalized-alpha scheme with controllable high-frequency dissipation via spectral_radius in [0, 1].
    - Interactive force perturbations: instantaneous point forces, moving surgical instruments, and harmonic excitations.
    - Decoupled real-time time stepper: physics time step dt is decoupled from rendering frame rate.
    - Live 3D visualization support: in-place PyVista/VTK vertex array mutation (polydata.points = sim.deformed_nodes) for non-blocking 60+ FPS rendering.
    - Comprehensive performance profiling: tracks wall-clock speed ratio (Sim Time / Real Time), average ms per step, and instantaneous energies.

    Parameters:
    -----------
    domain : Domain
        Physical domain (SolidDomain, ShellDomain, CoupledSolidShellDomain).
    damping : RayleighDamping, optional
        Rayleigh mass/stiffness proportional damping C = alpha * M + beta * K.
    dirichlet : DirichletManager or list of DirichletBoundaryCondition, optional
        Prescribed essential displacement boundary conditions.
    dt : float, default=0.005
        Fixed physics simulation time increment in seconds (e.g., 0.005s = 200 Hz).
    spectral_radius : float, default=0.8
        High-frequency spectral radius rho_inf in [0.0, 1.0] for Generalized-alpha damping.
    f_ext_base : np.ndarray of shape (n_dofs,), optional
        Base external load vector (e.g. constant gravitational body force).
    """

    def __init__(
        self,
        domain: Domain,
        damping: RayleighDamping | None = None,
        dirichlet: DirichletManager | Sequence[DirichletBoundaryCondition] | None = None,
        dt: float = 0.005,
        spectral_radius: float = 0.8,
        f_ext_base: np.ndarray | None = None,
        nonlinear: bool = False,
        nr_max_iter: int = 1,
        nr_tol: float = 1e-3,
        modified_newton: bool = True,
    ) -> None:
        super().__init__()
        self.domain = domain
        self.nonlinear = bool(nonlinear)
        self.nr_max_iter = max(1, int(nr_max_iter))
        self.nr_tol = float(nr_tol)
        self.modified_newton = bool(modified_newton)

        if dt <= 0.0:
            raise ValueError(f"Time step dt must be positive, got {dt}")
        self.dt = float(dt)

        if not (0.0 <= spectral_radius <= 1.0):
            raise ValueError(
                f"spectral_radius must be in [0.0, 1.0], got {spectral_radius}"
            )
        self.spectral_radius = float(spectral_radius)

        # Generalized-alpha algorithmic parameters
        rho = self.spectral_radius
        self.alpha_m = float((2.0 * rho - 1.0) / (rho + 1.0))
        self.alpha_f = float(rho / (rho + 1.0))
        self.gamma = float(0.5 - self.alpha_m + self.alpha_f)
        self.beta = float(0.25 * (1.0 - self.alpha_m + self.alpha_f) ** 2)

        # Integration constants
        self._a0 = 1.0 / (self.beta * self.dt**2)
        self._a1 = self.gamma / (self.beta * self.dt)
        self._a2 = 1.0 / (self.beta * self.dt)
        self._a3 = 1.0 / (2.0 * self.beta) - 1.0
        self._a4 = self.gamma / self.beta - 1.0
        self._a5 = self.dt * 0.5 * (self.gamma / self.beta - 2.0)

        # Physical damping
        self.damping = (
            damping if damping is not None else RayleighDamping(alpha=0.0, beta=0.0)
        )

        # Dirichlet Boundary Management
        if dirichlet is None:
            self.dirichlet_manager = DirichletManager(self.domain.n_dofs)
        elif isinstance(dirichlet, DirichletManager):
            self.dirichlet_manager = dirichlet
        else:
            self.dirichlet_manager = DirichletManager(self.domain.n_dofs)
            for c in dirichlet:
                self.dirichlet_manager.add_constraint(c)

        self.free_dofs = self.dirichlet_manager.free_dofs
        self.p_dofs = self.dirichlet_manager.prescribed_dofs
        self.u_p = self.dirichlet_manager.prescribed_values

        if len(self.free_dofs) == 0:
            raise ValueError("All degrees of freedom are constrained. Nothing to solve.")

        # Assemble Mass, Stiffness, and Damping matrices
        self.M = self.domain.assemble_mass_matrix()
        self.K = self.domain.assemble_stiffness_matrix()
        self.C = self.damping.assemble_damping_matrix(self.M, self.K)

        # Partition matrices to free degrees of freedom
        self.M_ff = self.M[self.free_dofs, :][:, self.free_dofs].tocsr()
        self.K_ff = self.K[self.free_dofs, :][:, self.free_dofs].tocsr()
        self.C_ff = self.C[self.free_dofs, :][:, self.free_dofs].tocsr()

        # Build effective dynamic matrix K_eff
        # K_eff = (1 - alpha_f) * K_ff + (1 - alpha_m) * a0 * M_ff + (1 - alpha_f) * a1 * C_ff
        self.K_eff = (
            (1.0 - self.alpha_f) * self.K_ff
            + (1.0 - self.alpha_m) * self._a0 * self.M_ff
            + (1.0 - self.alpha_f) * self._a1 * self.C_ff
        ).tocsc()

        # Pre-factorize K_eff ONCE via SuperLU for ultra-fast O(N) back-substitutions
        self._solve_step = spspla.factorized(self.K_eff)

        # External loads
        self.f_ext_base = (
            np.asarray(f_ext_base, dtype=np.float64).copy()
            if f_ext_base is not None
            else np.zeros(self.domain.n_dofs, dtype=np.float64)
        )
        self.f_ext_dynamic = np.zeros(self.domain.n_dofs, dtype=np.float64)

        # Harmonic excitation parameters
        self._harmonic_sources: list[dict] = []

        # Kinematic state buffers (free DOFs)
        self.u_f = self.domain.u[self.free_dofs].copy()
        self.v_f = self.domain.v[self.free_dofs].copy()

        # Initial acceleration: M_ff * a0_f = F_ext_f - C_ff * v_f - K_ff * u_f
        f_init_f = (self.f_ext_base + self.f_ext_dynamic)[self.free_dofs]
        r0 = f_init_f - self.C_ff @ self.v_f - self.K_ff @ self.u_f
        self.a_f = spspla.spsolve(self.M_ff.tocsc(), r0)

        # Full global state buffers
        self.u_full = np.zeros(self.domain.n_dofs, dtype=np.float64)
        self.v_full = np.zeros(self.domain.n_dofs, dtype=np.float64)
        self.a_full = np.zeros(self.domain.n_dofs, dtype=np.float64)

        if len(self.p_dofs) > 0:
            self.u_full[self.p_dofs] = self.u_p
        self.u_full[self.free_dofs] = self.u_f
        self.v_full[self.free_dofs] = self.v_f
        self.a_full[self.free_dofs] = self.a_f

        # Simulation clock and performance telemetry
        self.current_time: float = 0.0
        self.step_count: int = 0
        self.total_wall_time: float = 0.0
        self._real_time_accumulator: float = 0.0

    def step(self) -> None:
        """
        Advances the physical simulation by one fixed time increment (dt) using
        the pre-factorized Generalized-alpha effective dynamic solver.
        """
        t0 = time.perf_counter()

        # Update harmonic excitations if configured
        if self._harmonic_sources:
            for src in self._harmonic_sources:
                dofs = src["dofs"]
                amp = src["amplitude"]
                freq = src["freq_hz"]
                phase = src["phase"]
                val = amp * np.sin(2.0 * np.pi * freq * self.current_time + phase)
                self.f_ext_dynamic[dofs] = val

        # Total external load on free DOFs
        f_ext_f = (self.f_ext_base + self.f_ext_dynamic)[self.free_dofs]

        if not self.nonlinear:
            # Generalized-alpha linear dynamic effective right-hand side vector:
            f_eff = (
                f_ext_f
                - self.alpha_f * (self.K_ff @ self.u_f)
                + self.M_ff
                @ (
                    (1.0 - self.alpha_m)
                    * (self._a0 * self.u_f + self._a2 * self.v_f + self._a3 * self.a_f)
                    - self.alpha_m * self.a_f
                )
                + self.C_ff
                @ (
                    (1.0 - self.alpha_f)
                    * (self._a1 * self.u_f + self._a4 * self.v_f + self._a5 * self.a_f)
                    - self.alpha_f * self.v_f
                )
            )

            # Solve for endpoint displacement u_{n+1} via O(N) SuperLU back-substitution
            u_next = self._solve_step(f_eff)
            a_next = (
                self._a0 * (u_next - self.u_f) - self._a2 * self.v_f - self._a3 * self.a_f
            )
            v_next = (
                self._a1 * (u_next - self.u_f) - self._a4 * self.v_f - self._a5 * self.a_f
            )
        else:
            # Generalized-alpha nonlinear dynamic Newton-Raphson iteration
            u_next = self.u_f.copy()
            f_ref_norm = max(float(np.linalg.norm(f_ext_f)), 1.0)

            for _ in range(self.nr_max_iter):
                a_next = (
                    self._a0 * (u_next - self.u_f) - self._a2 * self.v_f - self._a3 * self.a_f
                )
                v_next = (
                    self._a1 * (u_next - self.u_f) - self._a4 * self.v_f - self._a5 * self.a_f
                )

                u_mid_f = (1.0 - self.alpha_f) * u_next + self.alpha_f * self.u_f
                v_mid_f = (1.0 - self.alpha_f) * v_next + self.alpha_f * self.v_f
                a_mid_f = (1.0 - self.alpha_m) * a_next + self.alpha_m * self.a_f

                self.u_full[self.free_dofs] = u_mid_f
                self.v_full[self.free_dofs] = v_mid_f
                self.a_full[self.free_dofs] = a_mid_f
                self.domain.update_state(u=self.u_full, v=self.v_full, a=self.a_full)

                try:
                    f_int = self.domain.assemble_internal_forces()
                except ValueError:
                    # Inversion recovery: restore previous displacement and dissipate velocity
                    u_next = self.u_f.copy()
                    self.v_f *= 0.5
                    break

                r_eff = (
                    f_ext_f
                    - (self.M_ff @ a_mid_f)
                    - (self.C_ff @ v_mid_f)
                    - f_int[self.free_dofs]
                )

                if np.linalg.norm(r_eff) / f_ref_norm < self.nr_tol:
                    break

                if self.modified_newton:
                    du = self._solve_step(r_eff)
                else:
                    K_tan = self.domain.assemble_tangent_stiffness_matrix()
                    K_tan_ff = K_tan[self.free_dofs, :][:, self.free_dofs].tocsc()
                    K_eff_dyn = (
                        (1.0 - self.alpha_f) * K_tan_ff
                        + (1.0 - self.alpha_m) * self._a0 * self.M_ff
                        + (1.0 - self.alpha_f) * self._a1 * self.C_ff
                    ).tocsc()
                    du = spspla.spsolve(K_eff_dyn, r_eff)

                u_next += du

            a_next = (
                self._a0 * (u_next - self.u_f) - self._a2 * self.v_f - self._a3 * self.a_f
            )
            v_next = (
                self._a1 * (u_next - self.u_f) - self._a4 * self.v_f - self._a5 * self.a_f
            )

        # Update state vectors
        self.u_f = u_next
        self.v_f = v_next
        self.a_f = a_next

        self.u_full[self.free_dofs] = self.u_f
        self.v_full[self.free_dofs] = self.v_f
        self.a_full[self.free_dofs] = self.a_f

        # Update domain state
        self.domain.update_state(u=self.u_full, v=self.v_full, a=self.a_full)

        # Increment simulation time and telemetry
        self.current_time += self.dt
        self.step_count += 1
        self.total_wall_time += time.perf_counter() - t0

    def advance_time(self, duration: float) -> int:
        """
        Advances the simulation forward by a given time duration in seconds.

        Parameters:
        -----------
        duration : float
            Total physical duration to advance in seconds.

        Returns:
        --------
        substeps : int
            Number of physics time steps executed.
        """
        if duration <= 0.0:
            return 0
        n_substeps = max(1, int(round(duration / self.dt)))
        for _ in range(n_substeps):
            self.step()
        return n_substeps

    def advance_wall_clock(
        self, elapsed_real_time: float, max_substeps: int = 50
    ) -> int:
        """
        Synchronizes physics simulation time with elapsed real-world wall-clock time.
        Accumulates elapsed real time and executes as many dt sub-steps as required.

        Parameters:
        -----------
        elapsed_real_time : float
            Elapsed real-world time in seconds since the last frame/call.
        max_substeps : int, default=50
            Maximum sub-steps allowed per call to prevent lag spikes ("spiral of death").

        Returns:
        --------
        substeps_taken : int
            Number of physics sub-steps executed in this synchronization update.
        """
        self._real_time_accumulator += elapsed_real_time
        substeps_taken = 0

        while self._real_time_accumulator >= self.dt and substeps_taken < max_substeps:
            self.step()
            self._real_time_accumulator -= self.dt
            substeps_taken += 1

        # Prevent runaway accumulation if lagging severely
        if self._real_time_accumulator > 2.0 * self.dt:
            self._real_time_accumulator = 0.0

        return substeps_taken

    def apply_perturbation_force(
        self, node_index: int, force_vector: Sequence[float]
    ) -> None:
        """
        Applies a localized external point force perturbation vector [Fx, Fy, Fz]
        at a specific node.

        Parameters:
        -----------
        node_index : int
            Target node index in the mesh.
        force_vector : sequence of 3 floats
            External force components [Fx, Fy, Fz] in Newtons.
        """
        dofs = np.array([node_index * 3, node_index * 3 + 1, node_index * 3 + 2])
        self.f_ext_dynamic[dofs] += np.asarray(force_vector, dtype=np.float64)

    def apply_perturbation_forces(
        self, node_indices: Sequence[int], forces: np.ndarray
    ) -> None:
        """
        Applies multiple external point force perturbations simultaneously.

        Parameters:
        -----------
        node_indices : sequence of int
            Target node indices.
        forces : np.ndarray of shape (N, 3)
            Force vectors [Fx, Fy, Fz] for each node.
        """
        forces_arr = np.asarray(forces, dtype=np.float64)
        for idx, node in enumerate(node_indices):
            dofs = np.array([node * 3, node * 3 + 1, node * 3 + 2])
            self.f_ext_dynamic[dofs] += forces_arr[idx]

    def clear_perturbation_forces(self) -> None:
        """Clears all interactive dynamic perturbation forces, resetting them to zero."""
        self.f_ext_dynamic.fill(0.0)

    def set_base_external_force(self, f_ext: np.ndarray) -> None:
        """Updates the permanent base external load vector (e.g., changing gravity or static loads)."""
        self.f_ext_base = np.asarray(f_ext, dtype=np.float64).copy()

    def add_harmonic_excitation(
        self,
        node_index: int,
        component: int,
        amplitude: float,
        frequency_hz: float,
        phase: float = 0.0,
    ) -> None:
        """
        Adds a continuous sinusoidal harmonic force excitation F(t) = A * sin(2*pi*f*t + phi)
        to a specific degree of freedom.

        Parameters:
        -----------
        node_index : int
            Node index where the excitation is applied.
        component : int
            DOF direction (0=X, 1=Y, 2=Z).
        amplitude : float
            Force amplitude in Newtons.
        frequency_hz : float
            Oscillation frequency in Hertz.
        phase : float, default=0.0
            Phase angle in radians.
        """
        dof = node_index * 3 + component
        self._harmonic_sources.append(
            {
                "dofs": dof,
                "amplitude": float(amplitude),
                "freq_hz": float(frequency_hz),
                "phase": float(phase),
            }
        )

    def clear_harmonic_excitations(self) -> None:
        """Removes all continuous harmonic excitation sources."""
        self._harmonic_sources.clear()

    def find_closest_node(
        self, point: Sequence[float], deformed: bool = True
    ) -> int:
        """
        Finds the mesh node index closest to the given 3D coordinates.

        Parameters:
        -----------
        point : sequence of 3 floats
            3D Cartesian coordinates [x, y, z].
        deformed : bool, default=True
            If True, searches current deformed spatial coordinates; if False, searches reference coordinates.

        Returns:
        --------
        node_index : int
            Index of the closest node.
        """
        pts = self.deformed_nodes if deformed else self.domain.mesh.nodes
        p = np.asarray(point, dtype=np.float64)
        dist_sq = np.sum((pts - p) ** 2, axis=1)
        return int(np.argmin(dist_sq))

    def update_pyvista_grid(
        self,
        grid: Any,
        plotter: Any | None = None,
        field: str = "Displacement_Y",
        auto_clim: bool = True,
        symmetric_clim: bool = True,
        min_clim: float = 0.5,
    ) -> None:
        """
        Updates an existing PyVista UnstructuredGrid in-place with current deformed coordinates
        and displacement scalars for zero-overhead real-time rendering.

        Parameters:
        -----------
        grid : pv.UnstructuredGrid
            Target PyVista grid previously created from the domain.
        plotter : pv.Plotter, optional
            Active PyVista Plotter instance. If provided and `auto_clim=True`, dynamically
            updates the scalar bar range so the color palette tracks instantaneous deflection.
        field : str, default="Displacement_Y"
            Active scalar field name to track for dynamic color mapping.
        auto_clim : bool, default=True
            Whether to automatically adjust the mapper scalar range when `plotter` is given.
        symmetric_clim : bool, default=True
            Whether to enforce symmetric scalar limits [-max_abs, +max_abs] centered at 0.0.
            Ideal for diverging colormaps (e.g. 'coolwarm') where 0.0 represents the neutral state.
        min_clim : float, default=0.5
            Minimum absolute extent for symmetric scalar bounds to prevent zero-width ranges.
        """
        u = self.displacement
        grid.points = self.deformed_nodes
        grid.point_data["Displacement"] = u
        grid.point_data["Displacement_X"] = u[:, 0]
        grid.point_data["Displacement_Y"] = u[:, 1]
        grid.point_data["Displacement_Z"] = u[:, 2]
        grid.point_data["Displacement_Mag"] = np.linalg.norm(u, axis=1)

        # Update synthetic component arrays if PyVista created them (e.g. from component=1)
        for comp_idx in range(3):
            comp_key = f"Displacement-{comp_idx}"
            if comp_key in grid.point_data:
                grid.point_data[comp_key] = u[:, comp_idx]

        # Trigger VTK pipeline modification notifications
        point_data = grid.GetPointData()
        if point_data is not None:
            active_scalars = point_data.GetScalars()
            if active_scalars is not None:
                active_scalars.Modified()
            point_data.Modified()

        # Update scalar bar range dynamically if plotter is provided
        if plotter is not None and auto_clim and field in grid.point_data:
            vals = grid.point_data[field]
            if symmetric_clim:
                max_abs = max(float(min_clim), float(np.max(np.abs(vals))))
                clim = [-max_abs, max_abs]
            else:
                val_min = float(np.min(vals))
                val_max = float(np.max(vals))
                if abs(val_max - val_min) < 1e-4:
                    val_min -= 0.5
                    val_max += 0.5
                clim = [val_min, val_max]
            try:
                plotter.update_scalar_bar_range(clim)
            except Exception:
                pass

    @property
    def deformed_nodes(self) -> np.ndarray:
        """Returns the current deformed 3D spatial node coordinates (X + u) of shape (N, 3)."""
        return self.domain.mesh.nodes + self.displacement

    @property
    def displacement(self) -> np.ndarray:
        """Returns the current displacement array of shape (N, 3) in millimeters."""
        return self.u_full.reshape(-1, 3)

    @property
    def velocity(self) -> np.ndarray:
        """Returns the current velocity array of shape (N, 3) in mm/s."""
        return self.v_full.reshape(-1, 3)

    @property
    def acceleration(self) -> np.ndarray:
        """Returns the current acceleration array of shape (N, 3) in mm/s²."""
        return self.a_full.reshape(-1, 3)

    @property
    def kinetic_energy(self) -> float:
        """Returns the instantaneous physical kinetic energy: E_k = 0.5 * v^T * M * v in Joules."""
        return 0.5 * float(self.v_f @ (self.M_ff @ self.v_f))

    @property
    def strain_energy(self) -> float:
        """Returns the instantaneous elastic strain energy in Joules."""
        if not self.nonlinear:
            return 0.5 * float(self.u_f @ (self.K_ff @ self.u_f))
        else:
            f_int = self.domain.assemble_internal_forces()
            return 0.5 * float(self.u_f @ f_int[self.free_dofs])

    @property
    def total_energy(self) -> float:
        """Returns the total mechanical energy: E_total = E_k + U in Joules."""
        return self.kinetic_energy + self.strain_energy

    def reset(self) -> None:
        """Resets the simulation to the initial resting state (t = 0.0, u = 0, v = 0, a = a0)."""
        self.u_f.fill(0.0)
        self.v_f.fill(0.0)

        f_init_f = (self.f_ext_base + self.f_ext_dynamic)[self.free_dofs]
        r0 = f_init_f - self.K_ff @ self.u_f
        self.a_f = spspla.spsolve(self.M_ff.tocsc(), r0)

        self.u_full.fill(0.0)
        self.v_full.fill(0.0)
        self.a_full.fill(0.0)

        if len(self.p_dofs) > 0:
            self.u_full[self.p_dofs] = self.u_p
        self.u_full[self.free_dofs] = self.u_f
        self.v_full[self.free_dofs] = self.v_f
        self.a_full[self.free_dofs] = self.a_f

        self.domain.update_state(u=self.u_full, v=self.v_full, a=self.a_full)
        self.current_time = 0.0
        self.step_count = 0
        self.total_wall_time = 0.0
        self._real_time_accumulator = 0.0

    def get_performance_stats(self) -> SimulationStats:
        """
        Calculates and returns performance metrics, speed ratios, and energy levels.
        """
        wall_time = self.total_wall_time
        steps = max(1, self.step_count)
        avg_ms = (wall_time / steps) * 1000.0 if wall_time > 0 else 0.0
        eff_hz = steps / wall_time if wall_time > 0 else 0.0
        speed_ratio = self.current_time / wall_time if wall_time > 0 else 0.0

        return SimulationStats(
            sim_time=self.current_time,
            wall_time=wall_time,
            steps_completed=self.step_count,
            average_step_ms=avg_ms,
            speed_ratio=speed_ratio,
            effective_hz=eff_hz,
            kinetic_energy=self.kinetic_energy,
            strain_energy=self.strain_energy,
            total_energy=self.total_energy,
        )

    def solve(
        self,
        duration: float = 1.0,
        f_ext: np.ndarray | None = None,
    ) -> SimulationStats:
        """
        Executes the dynamic simulation for a prescribed time duration (Analysis interface).

        Parameters:
        -----------
        duration : float, default=1.0
            Total simulation time in seconds to advance.
        f_ext : np.ndarray of shape (n_dofs,), optional
            Optional base external load vector to apply during the run.

        Returns:
        --------
        stats : SimulationStats
            Execution telemetry, speed ratio, and energy metrics.
        """
        if f_ext is not None:
            self.set_base_external_force(f_ext)
        self.advance_time(duration)
        return self.get_performance_stats()
