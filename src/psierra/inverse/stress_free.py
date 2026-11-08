"""
Stress-Free Reference Configuration Solver (Zero-Pressure / Zero-Gravity State).

Determines the undeformed, stress-free reference geometry X_0 from clinical
imaging scans or optical surface scans of pre-gravitationally sagged or
pre-stressed anatomical bodies (X_scan).

Mathematical Formulation:
-------------------------
In clinical imaging (e.g., MRI, CT) and physical measurements, organs and soft
tissues are scanned in an in vivo state pre-deformed by physiological boundary
loads (such as gravity g_scan, cerebrospinal fluid pressure, or cranial contact):
    X_scan = X_0 + u(X_0; L_scan)

If X_scan is erroneously taken as the reference (stress-free) configuration in a
subsequent surgical simulation, applying operative loads produces double-gravity
bias:
    x_biased = X_scan + u(X_scan; L_surgery) != X_0 + u(X_0; L_surgery)

To eliminate this bias, we formulate the backward fixed-point problem to find
the unknown stress-free geometry X_0 satisfying:
    F(X_0) := X_0 + u(X_0; L_scan) - X_scan = 0

Sellier Fixed-Point Iteration with Under-Relaxation:
---------------------------------------------------
Let X_0^(0) = X_scan (initial nominal estimate). At each iteration k >= 0:
1. Forward Equilibrium Solve:
       Compute displacement u^(k) = u(X_0^(k); L_scan) on candidate reference X_0^(k).
2. Spatial Configuration & Residual:
       x^(k) = X_0^(k) + u^(k)
       R^(k) = X_scan - x^(k) = X_scan - X_0^(k) - u^(k)
3. Backward Update with Relaxation omega in (0, 1]:
       X_0^(k+1) = X_0^(k) + omega * R^(k)

When omega = 1.0, this corresponds to the classical Sellier update:
       X_0^(k+1) = X_scan - u^(k)
For soft tissues undergoing moderate-to-large deflections, under-relaxation
omega in [0.5, 0.8] or adaptive Aitken Delta^2 dynamic relaxation guarantees
monotonic geometric convergence without element inversion.

References:
-----------
- Sellier, M. (2011). "An iterative method for the determination of the
  stress-free configuration of hyperelastic materials." Computer Methods in
  Applied Mechanics and Engineering, 200(49-52), 3381-3388.
- Govindjee, S., & Mihalic, P. A. (1996). "Computational methods for inverse
  finite elastostatics." Computer Methods in Applied Mechanics and Engineering,
  136(1-2), 47-57.
- Rajagopal, V., et al. (2007). "Determining the uncompressed breast geometry
  from MRI data for image-guided breast surgery." IEEE Transactions on Medical
  Imaging, 27(9), 1188-1198.
- Bilger, A., et al. (2014). "Biomechanical simulation of brain shift during
  neurosurgery." International Journal of Computer Assisted Radiology and Surgery.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Literal, Sequence

import numpy as np

from psierra.mesh.volumetric import VolumetricMesh
from psierra.domain.solid import SolidDomain
from psierra.boundary.dirichlet import DirichletConstraint, DirichletManager
from psierra.boundary.load import GravityLoad
from psierra.analysis.linear_static import LinearStaticAnalysis
from psierra.analysis.nonlinear_static import NonlinearStaticAnalysis


@dataclass(frozen=True)
class StressFreeResult:
    """
    Container for stress-free reference configuration solver results.

    Attributes:
    -----------
    reference_nodes : np.ndarray
        Recovered undeformed stress-free reference nodal coordinates X_0* of
        shape (N_nodes, 3).
    scan_nodes : np.ndarray
        Target pre-deformed scan configuration X_scan of shape (N_nodes, 3).
    predicted_scan_nodes : np.ndarray
        Spatial coordinates x* = X_0* + u* under scan load L_scan of shape
        (N_nodes, 3).
    displacement : np.ndarray
        Nodal displacement field u* from reference X_0* to predicted scan x*
        under L_scan. Stored as 1D array of shape (n_dofs,).
    residual_error : float
        Maximum nodal position error ||R||_inf = max_i ||R_i||_2 in mm.
    relative_error : float
        Relative Frobenius norm ||R||_F / ||X_scan||_F.
    rms_error : float
        Root-mean-square nodal position discrepancy sqrt(mean(||R_i||^2)) in mm.
    iterations : int
        Number of fixed-point iterations performed.
    converged : bool
        Whether residual error satisfied the specified tolerance.
    residual_history : list[float]
        History of maximum nodal residual error per iteration.
    wall_time_seconds : float
        Total solver wall-clock time in seconds.
    """

    reference_nodes: np.ndarray
    scan_nodes: np.ndarray
    predicted_scan_nodes: np.ndarray
    displacement: np.ndarray
    residual_error: float
    relative_error: float
    rms_error: float
    iterations: int
    converged: bool
    residual_history: list[float] = field(default_factory=list)
    wall_time_seconds: float = 0.0

    @property
    def displacement_3d(self) -> np.ndarray:
        """Nodal displacement reshaped to (N_nodes, 3)."""
        return self.displacement.reshape(-1, 3)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the discretized domain."""
        return len(self.reference_nodes)

    def create_reference_mesh(self, base_mesh: VolumetricMesh) -> VolumetricMesh:
        """
        Constructs a new VolumetricMesh with the recovered stress-free coordinates.

        Parameters:
        -----------
        base_mesh : VolumetricMesh
            Template volumetric mesh sharing identical topology and element connectivity.

        Returns:
        --------
        VolumetricMesh
            Deep copy of the base mesh with nodal coordinates set to X_0*.
        """
        if base_mesh.n_nodes != self.n_nodes:
            raise ValueError(
                f"Base mesh has {base_mesh.n_nodes} nodes; expected {self.n_nodes}."
            )
        mesh_copy = base_mesh.copy()
        mesh_copy.nodes = self.reference_nodes.copy()
        return mesh_copy

    def create_predicted_mesh(self, base_mesh: VolumetricMesh) -> VolumetricMesh:
        """
        Constructs a new VolumetricMesh with the predicted spatial coordinates under scan load.

        Parameters:
        -----------
        base_mesh : VolumetricMesh
            Template volumetric mesh sharing identical topology and element connectivity.

        Returns:
        --------
        VolumetricMesh
            Deep copy of the base mesh with nodal coordinates set to x*.
        """
        if base_mesh.n_nodes != self.n_nodes:
            raise ValueError(
                f"Base mesh has {base_mesh.n_nodes} nodes; expected {self.n_nodes}."
            )
        mesh_copy = base_mesh.copy()
        mesh_copy.nodes = self.predicted_scan_nodes.copy()
        return mesh_copy

    def summary(self) -> str:
        """Returns a formatted diagnostic summary string."""
        status = "Converged" if self.converged else "Did NOT Converge"
        return (
            f"StressFreeResult:\n"
            f"  • Status          : {status} in {self.iterations} iterations\n"
            f"  • Max Residual    : {self.residual_error:.4e} mm\n"
            f"  • RMS Error       : {self.rms_error:.4e} mm\n"
            f"  • Relative Error  : {self.relative_error * 100:.4f}%\n"
            f"  • Domain Nodes    : {self.n_nodes}\n"
            f"  • Solve Time      : {self.wall_time_seconds * 1e3:.2f} ms"
        )


class StressFreeConfigurationSolver:
    """
    Backward Fixed-Point Solver for Stress-Free Reference Configurations.

    Iteratively reconstructs the undeformed geometry X_0 from pre-sagged scans
    X_scan subjected to known physiological loads L_scan (Sellier 2011).

    Parameters:
    -----------
    domain : SolidDomain
        Physical continuum domain defining mesh topology, element formulation,
        and material constitutive law.
    load : GravityLoad, np.ndarray, or Callable[[SolidDomain], np.ndarray]
        External loading condition acting during scan acquisition (e.g., gravity).
    dirichlet : DirichletManager or list of DirichletConstraint
        Essential boundary conditions active during the scan.
    relaxation_factor : float, default=0.7
        Under-relaxation parameter omega in (0, 1].
    adaptive_relaxation : bool, default=False
        If True, employs dynamic Aitken Delta^2 relaxation across iterations.
    tolerance : float, default=1e-4
        Convergence tolerance on maximum nodal position residual (mm).
    max_iterations : int, default=25
        Maximum number of fixed-point iterations.
    analysis_type : Literal["linear", "nonlinear"], default="linear"
        Static forward solver type ("linear" or "nonlinear").
    nonlinear_kwargs : dict[str, Any], optional
        Additional kwargs passed to NonlinearStaticAnalysis.solve() (e.g. n_steps).
    """

    def __init__(
        self,
        domain: SolidDomain,
        load: GravityLoad | np.ndarray | Callable[[SolidDomain], np.ndarray],
        dirichlet: DirichletManager | Sequence[DirichletConstraint],
        relaxation_factor: float = 0.7,
        adaptive_relaxation: bool = False,
        tolerance: float = 1e-4,
        max_iterations: int = 25,
        analysis_type: Literal["linear", "nonlinear"] = "linear",
        nonlinear_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not (0.0 < relaxation_factor <= 1.0):
            raise ValueError(
                f"relaxation_factor must lie in (0, 1], got {relaxation_factor}."
            )
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be strictly positive, got {tolerance}.")
        if max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {max_iterations}."
            )
        if analysis_type not in ("linear", "nonlinear"):
            raise ValueError(
                f"analysis_type must be 'linear' or 'nonlinear', got '{analysis_type}'."
            )

        self._domain = domain
        self._load = load

        if isinstance(dirichlet, DirichletManager):
            self._dirichlet = dirichlet
        elif isinstance(dirichlet, Sequence):
            dm = DirichletManager(domain.mesh.n_dofs)
            for c in dirichlet:
                dm.add_constraint(c)
            self._dirichlet = dm
        else:
            raise TypeError(
                f"dirichlet must be DirichletManager or list of DirichletConstraint, got {type(dirichlet)}."
            )

        self._relaxation_factor = float(relaxation_factor)
        self._adaptive_relaxation = bool(adaptive_relaxation)
        self._tolerance = float(tolerance)
        self._max_iterations = int(max_iterations)
        self._analysis_type = analysis_type
        self._nonlinear_kwargs = nonlinear_kwargs.copy() if nonlinear_kwargs else {}

    @property
    def domain(self) -> SolidDomain:
        """Reference template domain."""
        return self._domain

    @property
    def relaxation_factor(self) -> float:
        """Current fixed relaxation factor omega."""
        return self._relaxation_factor

    @property
    def tolerance(self) -> float:
        """Convergence tolerance in mm."""
        return self._tolerance

    @property
    def max_iterations(self) -> int:
        """Maximum iteration count."""
        return self._max_iterations

    @property
    def analysis_type(self) -> str:
        """Static analysis formulation ('linear' or 'nonlinear')."""
        return self._analysis_type

    def _compute_external_load(self, active_domain: SolidDomain) -> np.ndarray:
        """Computes external force vector for the current reference domain."""
        if isinstance(self._load, GravityLoad):
            M = active_domain.assemble_mass_matrix()
            return self._load.compute_force_vector(mass_matrix=M)
        elif callable(self._load):
            return np.asarray(self._load(active_domain), dtype=np.float64)
        elif isinstance(self._load, np.ndarray):
            f = np.asarray(self._load, dtype=np.float64)
            if f.ndim != 1 or len(f) != active_domain.mesh.n_dofs:
                raise ValueError(
                    f"Load vector length ({len(f)}) does not match domain DOFs ({active_domain.mesh.n_dofs})."
                )
            return f
        else:
            raise TypeError(
                f"Unsupported load type: {type(self._load)}. Expected GravityLoad, ndarray, or callable."
            )

    def solve(
        self,
        scan_mesh_or_nodes: VolumetricMesh | np.ndarray,
        initial_guess: VolumetricMesh | np.ndarray | None = None,
        verbose: bool = False,
    ) -> StressFreeResult:
        """
        Executes the backward fixed-point iteration to recover the stress-free geometry.

        Parameters:
        -----------
        scan_mesh_or_nodes : VolumetricMesh or np.ndarray
            Target pre-deformed scan configuration X_scan. Either a VolumetricMesh
            or coordinate array of shape (N_nodes, 3).
        initial_guess : VolumetricMesh or np.ndarray, optional
            Initial estimate for reference coordinates X_0^(0). Defaults to X_scan.
        verbose : bool, default=False
            If True, prints progress telemetry per iteration.

        Returns:
        --------
        StressFreeResult
            Reconstructed reference geometry, convergence history, and error metrics.
        """
        t0 = time.perf_counter()

        # 1. Parse target scan coordinates X_scan
        if isinstance(scan_mesh_or_nodes, VolumetricMesh):
            X_scan = scan_mesh_or_nodes.nodes.copy()
        elif isinstance(scan_mesh_or_nodes, np.ndarray):
            X_scan = np.asarray(scan_mesh_or_nodes, dtype=np.float64).copy()
        else:
            raise TypeError(
                f"scan_mesh_or_nodes must be VolumetricMesh or np.ndarray, got {type(scan_mesh_or_nodes)}."
            )

        if X_scan.ndim != 2 or X_scan.shape[1] != 3:
            raise ValueError(
                f"scan coordinates must have shape (N_nodes, 3), got {X_scan.shape}."
            )
        if len(X_scan) != self._domain.mesh.n_nodes:
            raise ValueError(
                f"scan node count ({len(X_scan)}) does not match template domain ({self._domain.mesh.n_nodes})."
            )

        # 2. Parse initial guess X_0
        if initial_guess is None:
            X_0 = X_scan.copy()
        elif isinstance(initial_guess, VolumetricMesh):
            X_0 = initial_guess.nodes.copy()
        elif isinstance(initial_guess, np.ndarray):
            X_0 = np.asarray(initial_guess, dtype=np.float64).copy()
        else:
            raise TypeError(
                f"initial_guess must be VolumetricMesh, np.ndarray, or None, got {type(initial_guess)}."
            )

        if X_0.shape != X_scan.shape:
            raise ValueError(
                f"initial_guess shape {X_0.shape} does not match X_scan shape {X_scan.shape}."
            )

        # 3. Setup isolated working domain and forward solver
        work_mesh = self._domain.mesh.copy()
        work_mesh.nodes = X_0.copy()
        work_domain = SolidDomain(
            work_mesh, self._domain.material, self._domain.element
        )

        if self._analysis_type == "linear":
            forward_analysis = LinearStaticAnalysis(work_domain)
        else:
            forward_analysis = NonlinearStaticAnalysis(work_domain)

        # 4. Iteration state
        omega = self._relaxation_factor
        residual_history: list[float] = []
        R_prev: np.ndarray | None = None
        converged = False
        last_u: np.ndarray = np.zeros(self._domain.mesh.n_dofs, dtype=np.float64)
        last_x_pred: np.ndarray = X_0.copy()
        last_max_res: float = float("inf")
        last_rms_res: float = float("inf")
        last_rel_res: float = float("inf")

        # 5. Fixed-Point Loop
        for it in range(self._max_iterations):
            # A. Update domain reference geometry to candidate X_0
            work_domain.mesh.nodes = X_0.copy()
            work_domain.u.fill(0.0)

            # B. Assemble external loads for current geometry
            f_ext = self._compute_external_load(work_domain)

            # C. Forward static solve
            if self._analysis_type == "linear":
                res = forward_analysis.solve(
                    dirichlet=self._dirichlet,
                    f_ext=f_ext,
                    verbose=False,
                )
            else:
                default_nl = {"n_steps": 2, "nr_iterations": 25, "nr_tol": 1e-4}
                nl_args = {**default_nl, **self._nonlinear_kwargs}
                res = forward_analysis.solve(
                    dirichlet=self._dirichlet,
                    f_ext=f_ext,
                    verbose=False,
                    **nl_args,
                )

            u_disp = res.displacement.reshape(-1, 3)
            last_u = res.displacement.copy()

            # D. Spatial configuration and residual
            last_x_pred = X_0 + u_disp
            R = X_scan - last_x_pred  # shape (N, 3)

            # Nodal residual norms
            nodal_norms = np.linalg.norm(R, axis=1)
            last_max_res = float(np.max(nodal_norms))
            last_rms_res = float(np.sqrt(np.mean(nodal_norms**2)))
            last_rel_res = float(
                np.linalg.norm(R) / (np.linalg.norm(X_scan) + 1e-15)
            )
            residual_history.append(last_max_res)

            if verbose:
                print(
                    f"  Iter {it:02d}: Max Residual = {last_max_res:.4e} mm | "
                    f"RMS = {last_rms_res:.4e} mm | "
                    f"Rel = {last_rel_res * 100:.3f}% | omega = {omega:.3f}"
                )

            # E. Check convergence
            if last_max_res <= self._tolerance:
                converged = True
                break

            # F. Adaptive Aitken relaxation update (if enabled and it >= 1)
            if self._adaptive_relaxation and R_prev is not None:
                delta_R = R - R_prev
                denom = float(np.sum(delta_R**2))
                if denom > 1e-14:
                    omega_aitken = -omega * float(np.sum(R_prev * delta_R)) / denom
                    omega = float(np.clip(omega_aitken, 0.1, 1.0))

            R_prev = R.copy()

            # G. Backward reference update
            X_0 = X_0 + omega * R

        wall_time = time.perf_counter() - t0

        return StressFreeResult(
            reference_nodes=X_0,
            scan_nodes=X_scan,
            predicted_scan_nodes=last_x_pred,
            displacement=last_u,
            residual_error=last_max_res,
            relative_error=last_rel_res,
            rms_error=last_rms_res,
            iterations=len(residual_history),
            converged=converged,
            residual_history=residual_history,
            wall_time_seconds=wall_time,
        )
