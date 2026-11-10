from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import numpy as np
from rich import print

from psierra.domain.base import Domain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.boundary.contact import RigidContact, DeformableContact, SelfContact
from psierra.analysis.base import Analysis
from psierra.exchange.writer import SimulationWriter


@dataclass
class DynamicRelaxationResult:
    """Container for Dynamic Relaxation static equilibrium results."""

    displacement: np.ndarray  # Final displacement vector (n_dofs,)
    reactions: np.ndarray  # Global reaction force vector (n_dofs,)
    converged: bool  # Whether equilibrium was reached within max_steps
    steps_completed: int  # Total pseudo-time iterations taken
    residual_norm: float  # Final out-of-balance residual norm ||R_f||
    relative_residual: float  # Final relative residual ||R_f|| / ||F_ext_f||
    kinetic_energy: float  # Final kinetic energy
    kinetic_peaks: int  # Number of kinetic energy damping resets encountered
    residual_history: list[float] = field(default_factory=list)  # Residual norm history
    time_series_files: list[Path] = field(default_factory=list)  # Exported XDMF / PVD files
    saved_file: Path | None = None  # Saved primary visualization file (e.g. result.xdmf)


class DynamicRelaxationAnalysis(Analysis):
    """
    Dynamic Relaxation (DR) Explicit Static Equilibrium Solver.

    Solves the static equilibrium equations R(u) = F_ext - F_int(u) = 0 by transforming
    the problem into a fictitious, highly damped dynamic system:
        M_fict * u_ddot + C_fict * u_dot + F_int(u) = F_ext

    Key Advantages:
    - Matrix-free explicit vector formulation: Never forms, factorizes, or inverts global tangent stiffness K_t.
    - O(N) memory complexity and zero fill-in.
    - Unconditionally robust against tangent singularities (det(K_t) <= 0), material softening,
      extreme geometric distortions, and non-smooth contact chatter.
    - Works seamlessly on both linear elastic and nonlinear hyperelastic/continua domains.
    - Fictitious Gerschgorin-scaled diagonal mass:
        M_ii >= 0.25 * dt^2 * sum_j |K_ij| * safety_factor
    - Damping options:
        * 'kinetic': Parameter-free Barnes/Underwood kinetic energy peak damping (v <- 0 when dEk/dt < 0).
        * 'viscous': Mass-proportional artificial damping factor.
    """

    def __init__(
        self,
        domain: Domain,
        damping_type: Literal["kinetic", "viscous"] = "kinetic",
        safety_factor: float = 1.1,
        viscous_damping: float | None = None,
        dt: float = 1.0,
    ) -> None:
        """
        Initializes the Dynamic Relaxation solver.

        Parameters:
        -----------
        domain : Domain
            Physical continuum domain (Linear, Neo-Hookean, Mooney-Rivlin, Shell, etc.).
        damping_type : Literal["kinetic", "viscous"], default="kinetic"
            Damping mechanism. 'kinetic' is parameter-free and recommended.
        safety_factor : float, default=1.1
            Gerschgorin mass scaling safety factor (must be >= 1.0 to ensure CFL stability).
        viscous_damping : float, optional
            Mass-proportional damping factor c (used only when damping_type="viscous").
        dt : float, default=1.0
            Fictitious pseudo-time increment.
        """
        self.domain = domain
        self.damping_type = damping_type.lower()
        if self.damping_type not in ("kinetic", "viscous"):
            raise ValueError(
                f"damping_type must be 'kinetic' or 'viscous', got '{damping_type}'"
            )
        self.safety_factor = float(safety_factor)
        self.viscous_damping = viscous_damping
        self.dt = float(dt)

    def _compute_fictitious_diagonal_mass(
        self,
        dt: float,
        free_dofs: np.ndarray,
        p_dofs: np.ndarray,
        contact_list: list[Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the fictitious diagonal mass matrix using Gerschgorin's circle theorem.
        Ensures that every DOF satisfies the explicit central-difference stability limit:
            omega_max * dt <= 2  ==>  M_ii >= 0.25 * dt^2 * sum_j |K_ij| * safety_factor.
        """
        K0 = self.domain.assemble_stiffness_matrix()
        row_sums = np.asarray(np.abs(K0).sum(axis=1)).flatten()

        # Account for contact penalty stiffness to prevent CFL instability upon impact
        if contact_list:
            for c in contact_list:
                if hasattr(c, "k_penalty"):
                    kp = float(c.k_penalty)
                    for node in getattr(c, "slave_nodes", []):
                        row_sums[3 * node : 3 * node + 3] += kp
                elif hasattr(c, "evaluate_stiffness"):
                    try:
                        Kc = c.evaluate_stiffness(
                            current_nodes=self.domain.current_nodes,
                            n_dofs=self.domain.n_dofs,
                        )
                        row_sums += np.asarray(np.abs(Kc).sum(axis=1)).flatten()
                    except Exception:
                        pass

        max_row = float(row_sums.max()) if row_sums.size > 0 and row_sums.max() > 0 else 1.0
        min_diag = 1e-6 * max_row
        row_sums = np.maximum(row_sums, min_diag)

        m_diag = 0.25 * self.safety_factor * (dt**2) * row_sums
        inv_m = np.zeros_like(m_diag)
        inv_m[free_dofs] = 1.0 / m_diag[free_dofs]
        inv_m[p_dofs] = 0.0

        return m_diag, inv_m

    def solve(
        self,
        dirichlet: DirichletManager | list[DirichletConstraint],
        f_ext: np.ndarray | None = None,
        tol: float = 1e-5,
        max_steps: int = 5000,
        min_steps: int = 10,
        contact: Any | None = None,
        include_stresses: bool = False,
        smooth_to_nodes: bool = False,
        save_directory: Path | str | None = None,
        output_name: str = "result",
        format: str = "xdmf",
        save_interval: int | None = None,
        save_trajectory: bool = False,
        verbose: bool = True,
        writer: SimulationWriter | None = None,
    ) -> DynamicRelaxationResult:
        """
        Executes the Dynamic Relaxation static equilibrium procedure.

        Parameters:
        -----------
        dirichlet : DirichletManager or list of DirichletConstraint
            Essential boundary constraints fixing or prescribing displacements.
        f_ext : np.ndarray, optional
            External force vector of shape (n_dofs,). If None, treated as zero force.
        tol : float, default=1e-5
            Convergence tolerance for the relative out-of-balance residual norm.
        max_steps : int, default=5000
            Maximum pseudo-time iterations before stopping.
        min_steps : int, default=10
            Minimum pseudo-time iterations before declaring convergence.
        contact : Contact or list of Contact, optional
            Rigid, deformable, or self-contact pairs.
        include_stresses : bool, default=False
            Whether to evaluate element stresses during state output.
        smooth_to_nodes : bool, default=False
            Whether to project stresses onto nodes.
        save_directory : Path or str, optional
            Directory to stream visualization frames to or save final static state.
        output_name : str, default="result"
            Base filename for export files.
        format : str, default="xdmf"
            File format ('xdmf' or 'pvd').
        save_interval : int, optional
            Iteration stride between recorded animation frames when save_trajectory is True.
        save_trajectory : bool, default=False
            If True, records the full pseudo-time relaxation trajectory animation.
            If False (default), records ONLY the final converged static equilibrium state
            (matching LinearStaticAnalysis behavior in ParaView).
        verbose : bool, default=True
            Whether to print iteration progress table.
        writer : SimulationWriter, optional
            Pre-configured external simulation writer.
        """
        self._validate_count("max_steps", max_steps, minimum=1)
        self._validate_count("min_steps", min_steps, minimum=1)
        self._validate_scalar("tol", tol)
        record_trajectory = save_trajectory or (
            save_interval is not None and save_interval > 0
        )
        if record_trajectory:
            stride = (
                save_interval
                if (save_interval is not None and save_interval > 0)
                else 50
            )
            self._validate_count("save_interval", stride, minimum=1)
        else:
            stride = 50

        # 1. Parse Boundary Conditions
        if isinstance(dirichlet, DirichletManager):
            manager = dirichlet
        else:
            manager = DirichletManager(self.domain.n_dofs)
            for c in dirichlet:
                manager.add_constraint(c)

        free_dofs = manager.free_dofs
        p_dofs = manager.prescribed_dofs
        u_p = manager.prescribed_values

        if len(free_dofs) == 0:
            raise ValueError("All degrees of freedom are constrained. Nothing to solve.")

        # 2. Setup External Force Vector
        if f_ext is None:
            f_ext_vec = np.zeros(self.domain.n_dofs, dtype=np.float64)
        else:
            f_ext_vec = np.asarray(f_ext, dtype=np.float64).ravel()
            if f_ext_vec.shape[0] != self.domain.n_dofs:
                raise ValueError(
                    f"f_ext length {f_ext_vec.shape[0]} does not match domain DOFs {self.domain.n_dofs}."
                )

        ref_force_norm = float(np.linalg.norm(f_ext_vec[free_dofs]))
        if ref_force_norm < 1e-12:
            ref_force_norm = 1.0

        # 3. Setup Contact Pairs
        if contact is None:
            if hasattr(self.domain, "contacts"):
                contact_list = list(self.domain.contacts)
            else:
                contact_list = []
        elif isinstance(contact, (RigidContact, DeformableContact, SelfContact)):
            contact_list = [contact]
        else:
            contact_list = list(contact)

        # 4. Initialize Coordinates and State
        dt = self.dt
        m_diag, inv_m = self._compute_fictitious_diagonal_mass(
            dt, free_dofs, p_dofs, contact_list=contact_list
        )

        u = self.domain.u.copy()
        u[p_dofs] = u_p
        v = np.zeros(self.domain.n_dofs, dtype=np.float64)
        self.domain.update_state(u=u, v=v)

        # 5. Simulation Writer Initialization (Trajectory Mode)
        save_dir = Path(save_directory) if save_directory is not None else None
        created_internal_writer = False
        if record_trajectory:
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

        # 6. Initial Force and Acceleration at step 0
        f_int = self.domain.assemble_internal_forces()
        f_contact = self._evaluate_contact_forces(contact_list, v)
        residual = f_ext_vec + f_contact - f_int
        a = inv_m * residual
        a[p_dofs] = 0.0

        res_norm = float(np.linalg.norm(residual[free_dofs]))
        rel_res = res_norm / ref_force_norm

        if verbose:
            print("[bold cyan]Dynamic Relaxation Analysis (Explicit Static Equilibrium)[/bold cyan]")
            print(
                f"Total DOFs: {self.domain.n_dofs} | Free DOFs: {len(free_dofs)} | Max Steps: {max_steps} | "
                f"Damping: {self.damping_type} | Tol: {tol:.1e}\n"
            )

        # Half-step velocity initialization
        if self.damping_type == "kinetic":
            v_half = 0.5 * dt * a
        else:
            c = self.viscous_damping if self.viscous_damping is not None else 0.4
            c1 = (2.0 - c * dt) / (2.0 + c * dt)
            c2 = (2.0 * dt) / (2.0 + c * dt)
            v_half = 0.5 * c2 * a

        v_half[p_dofs] = 0.0
        prev_Ek = 0.0
        kinetic_peaks = 0
        residual_history: list[float] = [res_norm]
        converged = False
        step = 0

        # 7. Explicit Pseudo-Time Relaxation Loop
        for step in range(1, max_steps + 1):
            u_prev = u.copy()
            a_prev = a.copy()

            # Advance displacement: u^{n+1} = u^n + dt * v^{n+1/2}
            u = u + dt * v_half
            u[p_dofs] = u_p
            self.domain.update_state(u=u, v=v_half)

            # Evaluate static residual at new coordinates
            try:
                f_int = self.domain.assemble_internal_forces()
            except (ValueError, FloatingPointError):
                # Backtrack on element inversion
                u = 0.5 * (u + u_prev)
                self.domain.update_state(u=u, v=np.zeros_like(v))
                f_int = self.domain.assemble_internal_forces()

            f_contact = self._evaluate_contact_forces(contact_list, v_half)
            residual = f_ext_vec + f_contact - f_int
            a = inv_m * residual
            a[p_dofs] = 0.0

            res_norm = float(np.linalg.norm(residual[free_dofs]))
            rel_res = res_norm / ref_force_norm
            residual_history.append(res_norm)

            # Advance velocity and compute kinetic energy
            if self.damping_type == "kinetic":
                v_next = v_half + dt * a
                v_next[p_dofs] = 0.0
                v_mid = 0.5 * (v_half + v_next)
                Ek = 0.5 * float(np.sum(m_diag[free_dofs] * (v_mid[free_dofs] ** 2)))

                # Peak kinetic energy detection
                if step > 1 and Ek < prev_Ek:
                    kinetic_peaks += 1
                    # Underwood peak displacement interpolation
                    u_peak = u - 1.5 * dt * v_half + 0.5 * (dt**2) * a_prev
                    u_peak[p_dofs] = u_p

                    # Safeguard peak update
                    try:
                        self.domain.update_state(u=u_peak, v=np.zeros_like(v))
                        f_int_test = self.domain.assemble_internal_forces()
                        if np.all(np.isfinite(f_int_test)):
                            u = u_peak
                            f_int = f_int_test
                    except (ValueError, FloatingPointError):
                        u = u_prev
                        self.domain.update_state(u=u, v=np.zeros_like(v))
                        f_int = self.domain.assemble_internal_forces()

                    f_contact = self._evaluate_contact_forces(contact_list, np.zeros_like(v))
                    residual = f_ext_vec + f_contact - f_int
                    a = inv_m * residual
                    a[p_dofs] = 0.0
                    res_norm = float(np.linalg.norm(residual[free_dofs]))
                    rel_res = res_norm / ref_force_norm

                    # Restart with zero velocity from peak
                    v_next = 0.5 * dt * a
                    v_next[p_dofs] = 0.0
                    prev_Ek = 0.0
                else:
                    prev_Ek = Ek

                v_half = v_next

            else:
                # Viscous mass-proportional damping update
                v_half = c1 * v_half + c2 * a
                v_half[p_dofs] = 0.0
                Ek = 0.5 * float(np.sum(m_diag[free_dofs] * (v_half[free_dofs] ** 2)))

            # Logging
            if verbose and (
                (record_trajectory and step % stride == 0)
                or step == 1
                or step % 50 == 0
            ):
                print(
                    f"  Step {step:5d} | ||R|| = {res_norm:11.4e} | Rel = {rel_res:11.4e} | "
                    f"Ek = {Ek:11.4e} | Peaks = {kinetic_peaks:3d}"
                )

            # Stream animation frame if writer active and trajectory requested
            if record_trajectory and writer is not None and step % stride == 0:
                writer.add_timestep(time=float(step), domain=self.domain, step_index=step)

            # Check convergence
            if step >= min_steps and rel_res <= tol:
                converged = True
                break

        # 8. Finalize Equilibrium State
        v_final = np.zeros(self.domain.n_dofs, dtype=np.float64)
        self.domain.update_state(u=u, v=v_final, a=np.zeros_like(v_final))
        reactions = f_int - f_ext_vec

        if record_trajectory:
            if writer is not None:
                writer.add_timestep(time=float(step), domain=self.domain, step_index=step)
                if created_internal_writer:
                    writer.close()
                elif hasattr(writer, "write"):
                    writer.write()
                exported_files = getattr(writer, "exported_files", [])
            else:
                exported_files = []
        else:
            # Default static solve: save converged equilibrium state (matching LinearStaticAnalysis)
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

        saved_file = exported_files[0] if exported_files else None

        if verbose:
            status_color = "bold green" if converged else "bold yellow"
            status_text = "converged" if converged else "reached max steps without full convergence"
            print(
                f"[{status_color}]✓ Dynamic Relaxation {status_text} in {step} steps "
                f"(Rel Res = {rel_res:.2e}, Kinetic Peaks = {kinetic_peaks}).[/{status_color}]\n"
            )

        return DynamicRelaxationResult(
            displacement=u.copy(),
            reactions=reactions,
            converged=converged,
            steps_completed=step,
            residual_norm=res_norm,
            relative_residual=rel_res,
            kinetic_energy=Ek,
            kinetic_peaks=kinetic_peaks,
            residual_history=residual_history,
            time_series_files=exported_files,
            saved_file=saved_file,
        )

    def _evaluate_contact_forces(
        self, contact_list: list[Any], velocities: np.ndarray
    ) -> np.ndarray:
        """Evaluates contact interface reaction forces across active contact objects."""
        if not contact_list:
            return np.zeros(self.domain.n_dofs, dtype=np.float64)
        return sum(
            (
                c.evaluate_forces(
                    current_nodes=self.domain.current_nodes,
                    velocities=velocities,
                    n_dofs=self.domain.n_dofs,
                )
                for c in contact_list
            ),
            np.zeros(self.domain.n_dofs, dtype=np.float64),
        )
