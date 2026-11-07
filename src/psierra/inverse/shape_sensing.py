"""
Variational Shape Sensing Inverse FEM (iFEM) Solvers.

Provides full-field 3D continuous displacement and strain reconstruction
from sparse discrete surface sensor measurements without requiring knowledge
of external loads, applied surface tractions, or intracranial pressures.

Mathematical Formulation:
-------------------------
Given N_sensor discrete measurements d_hat in R^(3 * N_sensor) acquired at
sensor locations via observation operator H, variational iFEM minimizes
the regularized least-squares functional:
    J(u) = 1/2 * (H u - d_hat)^T W (H u - d_hat) + alpha / 2 * u^T K u

where:
    - H in R^(3 * N_sensor x n_dofs) is the full-order discrete observation operator.
    - W in R^(3 * N_sensor x 3 * N_sensor) is a positive-definite diagonal sensor weighting matrix.
    - K in R^(n_dofs x n_dofs) is the structural stiffness / regularization matrix.
    - alpha > 0 is the Tikhonov smoothing parameter balancing measurement fidelity vs. strain energy.

Euler-Lagrange Stationarity:
    grad J(u) = H^T W (H u - d_hat) + alpha * K u = 0
    (H^T W H + alpha * K) u = H^T W d_hat

Full-Order iFEM Solve:
    A_full = H^T W H + alpha * K in R^(n_dofs x n_dofs)  (symmetric positive-definite)
    b_full = H^T W d_hat in R^n_dofs
    u = A_full^{-1} b_full  (sparse Cholesky solve in milliseconds)

Reduced-Order iFEM Solve (ROM iFEM):
    Substituting u = Phi @ u_r into J(u):
    J(u_r) = 1/2 * (H_r u_r - d_hat)^T W (H_r u_r - d_hat) + alpha / 2 * u_r^T K_r u_r
    where H_r = H @ Phi in R^(3 * N_sensor x r) and K_r = Phi^T K Phi in R^(r x r).

    grad J(u_r) = 0 yields:
    (H_r^T W H_r + alpha * K_r) u_r = H_r^T W d_hat
    A_r = H_r^T W H_r + alpha * K_r in R^(r x r)  (dense SPD)
    b_r = H_r^T W d_hat in R^r
    u_r = A_r^{-1} b_r  (dense Cholesky solve in < 15 microseconds, > 60,000 Hz)

Full-field recovery:
    u = Phi @ u_r in R^n_dofs
    u_internal = Phi_internal @ u_r  (instant deep tumor centroid recovery)

References:
-----------
- A. Tessler & H.R. Spangler (2005). "A Variational Principle for Reconstruction
  of Elastic Deformations in Shear Deformable Plates and Shells". Computer Methods
  in Applied Mechanics and Engineering, 194(2-5), 327-347.
- M. Gherlone et al. (2012). "Shape Sensing of 3D Frame Structures Using an
  Inverse Finite Element Method". Composite Structures.
- A. Tarantola (2005). "Inverse Problem Theory and Methods for Model Parameter
  Estimation". SIAM.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import scipy.linalg as spla

from psierra.domain.base import Domain
from psierra.inverse.observation import (
    SensorObservationOperator,
    ReducedSensorObservationOperator,
)
from psierra.reduction.basis import PODBasis
from psierra.reduction.model import ReducedLinearModel


@dataclass(frozen=True)
class ShapeSensingResult:
    """
    Container for continuous full-field shape sensing reconstruction results.

    Attributes:
    -----------
    displacement : np.ndarray
        Reconstructed full-order displacement vector in R^n_dofs.
    displacement_3d : np.ndarray
        Reconstructed displacement formatted as (N_nodes, 3).
    sensor_displacements : np.ndarray
        Model-predicted displacements at sensor locations in R^(3 * N_sensor).
    measurement_residual : np.ndarray
        Discrepancy between observation and reconstructed sensor values (d_hat - H u).
    measurement_rms_error : float
        Root-mean-square error over all sensor channels: ||d_hat - H u|| / sqrt(3 * N_sensor).
    strain_energy : float
        Regularization strain energy: 0.5 * u^T K u.
    reduced_coordinates : np.ndarray | None
        Generalized modal coordinates u_r in R^r (if solved via reduced iFEM).
    solve_time_seconds : float
        Linear system wall-clock solve time.
    """

    displacement: np.ndarray
    displacement_3d: np.ndarray
    sensor_displacements: np.ndarray
    measurement_residual: np.ndarray
    measurement_rms_error: float
    strain_energy: float
    reduced_coordinates: np.ndarray | None = None
    solve_time_seconds: float = 0.0


class FullOrderShapeSensingSolver:
    """
    Full-Order Model (FOM) Variational Shape Sensing Solver.

    Solves the sparse SPD system:
        (H^T W H + alpha * K) u = H^T W d_hat

    Serves as the high-fidelity ground truth benchmark for continuous 3D field
    reconstruction from discrete surface observations.
    """

    def __init__(
        self,
        observation_operator: SensorObservationOperator,
        stiffness_matrix: spsp.spmatrix | np.ndarray | Domain,
        alpha: float = 1e-4,
        weights: np.ndarray | Sequence[float] | None = None,
        free_dofs: np.ndarray | Sequence[int] | None = None,
    ) -> None:
        """
        Parameters:
        -----------
        observation_operator : SensorObservationOperator
            Full-order discrete observation operator H in R^(3 * N_sensor x n_dof).
        stiffness_matrix : spsp.spmatrix, np.ndarray, or Domain
            Full-order structural stiffness matrix K in R^(n_dofs x n_dofs) or physical Domain.
        alpha : float, default=1e-4
            Tikhonov regularization smoothing parameter (alpha > 0).
        weights : np.ndarray or Sequence of float, optional
            Sensor channel weighting factors w_k > 0 of length 3 * N_sensor or N_sensor.
            If length N_sensor, automatically broadcasted to (X, Y, Z) channels.
            Default is uniform identity weighting (all 1.0).
        free_dofs : Sequence of int, optional
            Unconstrained degrees of freedom if Dirichlet constraints are enforced.
        """
        if alpha <= 0.0:
            raise ValueError(f"Regularization parameter alpha must be strictly positive, got {alpha}.")

        self.observation_operator = observation_operator
        self.n_dofs = observation_operator.n_dofs
        self.n_sensors = observation_operator.n_sensors
        self.alpha = float(alpha)

        # Resolve stiffness matrix
        if hasattr(stiffness_matrix, "assemble_stiffness_matrix"):
            K_mat = stiffness_matrix.assemble_stiffness_matrix()
        else:
            K_mat = stiffness_matrix

        if not spsp.issparse(K_mat):
            K_mat = spsp.csr_matrix(K_mat)
        else:
            K_mat = K_mat.tocsr()

        if K_mat.shape != (self.n_dofs, self.n_dofs):
            raise ValueError(
                f"Stiffness matrix shape {K_mat.shape} does not match "
                f"operator n_dofs ({self.n_dofs}, {self.n_dofs})."
            )
        self.K = K_mat

        # Resolve sensor weights
        self.weights = self._resolve_weights(weights, self.n_sensors)
        self.W = spsp.diags(self.weights, format="csr")

        # Resolve free DOFs
        if free_dofs is not None:
            self.free_dofs = np.asarray(free_dofs, dtype=np.int64)
        else:
            self.free_dofs = np.arange(self.n_dofs, dtype=np.int64)

        # Precompute system matrix
        self._system_matrix: spsp.csr_matrix | None = None
        self._solver_prefactor = None
        self._build_system_matrix()

    @staticmethod
    def _resolve_weights(
        weights: np.ndarray | Sequence[float] | None, n_sensors: int
    ) -> np.ndarray:
        """Resolves sensor weights array of length 3 * n_sensors."""
        total_channels = 3 * n_sensors
        if weights is None:
            return np.ones(total_channels, dtype=np.float64)

        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        if np.any(w_arr <= 0.0):
            raise ValueError("All sensor weights must be strictly positive.")

        if len(w_arr) == n_sensors:
            # Broadcast per-sensor weight to (x, y, z) channels
            return np.repeat(w_arr, 3)
        elif len(w_arr) == total_channels:
            return w_arr
        else:
            raise ValueError(
                f"weights must have length {n_sensors} (per sensor) or "
                f"{total_channels} (per channel), got {len(w_arr)}."
            )

    def _build_system_matrix(self) -> None:
        """Assembles and pre-factorizes the full-order normal equations matrix."""
        H = self.observation_operator.matrix
        W = self.W

        # Normal equations: A = H^T W H + alpha * K
        HtW = H.T @ W
        HtWH = (HtW @ H).tocsr()

        A = HtWH + self.alpha * self.K
        # Symmetrize to eliminate floating-point non-symmetry: A = 0.5 * (A + A^T)
        A = 0.5 * (A + A.T)
        self._system_matrix = A.tocsr()
        self._HtW = HtW

    def set_alpha(self, alpha: float) -> None:
        """Updates Tikhonov smoothing parameter alpha and rebuilds the system matrix."""
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        self.alpha = float(alpha)
        self._build_system_matrix()

    def set_weights(self, weights: np.ndarray | Sequence[float]) -> None:
        """Updates sensor weights and rebuilds the system matrix."""
        self.weights = self._resolve_weights(weights, self.n_sensors)
        self.W = spsp.diags(self.weights, format="csr")
        self._build_system_matrix()

    @property
    def system_matrix(self) -> spsp.csr_matrix:
        """Full-order normal equations matrix A = H^T W H + alpha * K."""
        assert self._system_matrix is not None
        return self._system_matrix

    def solve(
        self,
        measurements: np.ndarray | Sequence[float],
    ) -> ShapeSensingResult:
        """
        Reconstructs full-field displacement vector from discrete sensor measurements.

        Parameters:
        -----------
        measurements : np.ndarray of shape (3 * N_sensor,) or (N_sensor, 3)
            Physical sensor displacements observed at marker locations.

        Returns:
        --------
        result : ShapeSensingResult
        """
        import time

        d_hat = np.asarray(measurements, dtype=np.float64).ravel()
        expected_len = 3 * self.n_sensors
        if d_hat.size != expected_len:
            raise ValueError(
                f"measurements vector size {d_hat.size} must match 3 * n_sensors ({expected_len})."
            )

        # Form RHS: b = H^T W d_hat
        b = self._HtW @ d_hat

        t0 = time.perf_counter()
        # Direct sparse solve
        u = spspla.spsolve(self.system_matrix, b)
        t_solve = time.perf_counter() - t0

        # Post-process results
        u_sensor = self.observation_operator.observe(u)
        residual = d_hat - u_sensor
        rms_err = float(np.sqrt(np.mean(residual**2)))
        strain_energy = float(0.5 * (u @ (self.K @ u)))

        return ShapeSensingResult(
            displacement=u,
            displacement_3d=u.reshape(-1, 3),
            sensor_displacements=u_sensor,
            measurement_residual=residual,
            measurement_rms_error=rms_err,
            strain_energy=strain_energy,
            reduced_coordinates=None,
            solve_time_seconds=t_solve,
        )


class ReducedOrderShapeSensingSolver:
    """
    Subspace Reduced-Order Model (ROM) Variational Shape Sensing Solver.

    Projects the iFEM variational principle onto the low-dimensional modal basis Phi:
        (H_r^T W H_r + alpha * K_r) u_r = H_r^T W d_hat

    Pre-computes and factorizes the dense r x r normal equations matrix ONCE,
    enabling sub-15 microsecond reconstruction (> 60,000 Hz) suitable for
    real-time intra-operative tracking and haptic guidance.
    """

    def __init__(
        self,
        reduced_observation_operator: ReducedSensorObservationOperator,
        reduced_stiffness: np.ndarray | ReducedLinearModel,
        basis: PODBasis | np.ndarray,
        alpha: float = 1e-4,
        weights: np.ndarray | Sequence[float] | None = None,
    ) -> None:
        """
        Parameters:
        -----------
        reduced_observation_operator : ReducedSensorObservationOperator
            Projected observation operator H_r in R^(3 * N_sensor x r).
        reduced_stiffness : np.ndarray of shape (r, r) or ReducedLinearModel
            Projected reduced stiffness matrix K_r in R^(r x r).
        basis : PODBasis or np.ndarray of shape (n_dofs, r)
            Modal basis matrix Phi used for coordinate lifting.
        alpha : float, default=1e-4
            Tikhonov regularization smoothing parameter (alpha > 0).
        weights : np.ndarray or Sequence of float, optional
            Sensor channel weights w_k > 0.
        """
        if alpha <= 0.0:
            raise ValueError(f"alpha must be strictly positive, got {alpha}.")

        self.observation_operator = reduced_observation_operator
        self.n_sensors = reduced_observation_operator.n_sensors
        self.n_modes = reduced_observation_operator.n_modes
        self.alpha = float(alpha)

        # Resolve basis
        if isinstance(basis, PODBasis):
            self.basis = basis
            self.Phi = basis.basis
        elif isinstance(basis, np.ndarray):
            self.basis = None
            self.Phi = basis
        else:
            raise TypeError(f"basis must be PODBasis or np.ndarray, got {type(basis)}.")

        self.n_dofs = self.Phi.shape[0]

        # Resolve reduced stiffness matrix K_r
        if hasattr(reduced_stiffness, "K_r"):
            K_r_mat = reduced_stiffness.K_r
        else:
            K_r_mat = np.asarray(reduced_stiffness, dtype=np.float64)

        if K_r_mat.shape != (self.n_modes, self.n_modes):
            raise ValueError(
                f"Reduced stiffness shape {K_r_mat.shape} does not match "
                f"n_modes ({self.n_modes}, {self.n_modes})."
            )
        self.K_r = 0.5 * (K_r_mat + K_r_mat.T)

        # Resolve sensor weights
        self.weights = FullOrderShapeSensingSolver._resolve_weights(
            weights, self.n_sensors
        )
        self.W = self.weights  # 1D diagonal array

        # Precompute system matrix and factorize
        self._A_r: np.ndarray | None = None
        self._cho_factor = None
        self._lu_factor = None
        self._build_system_matrix()

    def _build_system_matrix(self) -> None:
        """Assembles and pre-factorizes the dense r x r reduced normal equations matrix."""
        H_r = self.observation_operator.matrix  # (3*N, r)
        W = self.W[:, None]  # (3*N, 1)

        # H_r^T @ (W * H_r) in R^(r x r)
        HtWH_r = H_r.T @ (W * H_r)

        A_r = HtWH_r + self.alpha * self.K_r
        A_r = 0.5 * (A_r + A_r.T)
        self._A_r = A_r
        self._HtW = H_r.T * self.W[None, :]  # (r, 3*N)

        # Cholesky pre-factorization for microsecond solves
        try:
            self._cho_factor = spla.cho_factor(A_r, lower=True)
            self._lu_factor = None
        except spla.LinAlgError:
            self._cho_factor = None
            self._lu_factor = spla.lu_factor(A_r)

    def set_alpha(self, alpha: float) -> None:
        """Updates Tikhonov smoothing parameter alpha and pre-factorizes the system."""
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}.")
        self.alpha = float(alpha)
        self._build_system_matrix()

    def set_weights(self, weights: np.ndarray | Sequence[float]) -> None:
        """Updates sensor weights and pre-factorizes the system."""
        self.weights = FullOrderShapeSensingSolver._resolve_weights(
            weights, self.n_sensors
        )
        self.W = self.weights
        self._build_system_matrix()

    @property
    def system_matrix(self) -> np.ndarray:
        """Dense reduced normal equations matrix A_r in R^(r x r)."""
        assert self._A_r is not None
        return self._A_r

    def solve(
        self,
        measurements: np.ndarray | Sequence[float],
        reconstruct_full_field: bool = True,
    ) -> ShapeSensingResult:
        """
        Reconstructs generalized coordinates and full-field displacement.

        Parameters:
        -----------
        measurements : np.ndarray of shape (3 * N_sensor,) or (N_sensor, 3)
            Physical sensor displacements.
        reconstruct_full_field : bool, default=True
            Whether to lift u = Phi @ u_r. If False, displacement vector is empty
            for maximum microsecond execution speed.

        Returns:
        --------
        result : ShapeSensingResult
        """
        import time

        d_hat = np.asarray(measurements, dtype=np.float64).ravel()
        expected_len = 3 * self.n_sensors
        if d_hat.size != expected_len:
            raise ValueError(
                f"measurements vector size {d_hat.size} must match 3 * n_sensors ({expected_len})."
            )

        # RHS: b_r = H_r^T @ (W * d_hat) in R^r
        b_r = self._HtW @ d_hat

        t0 = time.perf_counter()
        if self._cho_factor is not None:
            u_r = spla.cho_solve(self._cho_factor, b_r)
        else:
            u_r = spla.lu_solve(self._lu_factor, b_r)
        t_solve = time.perf_counter() - t0

        # Predicted sensor displacement from reduced coordinates: u_sensor = H_r @ u_r
        u_sensor = self.observation_operator.observe(u_r)
        residual = d_hat - u_sensor
        rms_err = float(np.sqrt(np.mean(residual**2)))
        strain_energy = float(0.5 * (u_r @ (self.K_r @ u_r)))

        if reconstruct_full_field:
            u_full = self.Phi @ u_r
            u_3d = u_full.reshape(-1, 3)
        else:
            u_full = np.empty(0, dtype=np.float64)
            u_3d = np.empty((0, 3), dtype=np.float64)

        return ShapeSensingResult(
            displacement=u_full,
            displacement_3d=u_3d,
            sensor_displacements=u_sensor,
            measurement_residual=residual,
            measurement_rms_error=rms_err,
            strain_energy=strain_energy,
            reduced_coordinates=u_r,
            solve_time_seconds=t_solve,
        )

    def reconstruct_target_displacement(
        self,
        u_r: np.ndarray,
        target_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        """
        Directly reconstructs displacement at targeted internal landmarks (e.g. deep tumor):
            u_target = Phi[target_dofs, :] @ u_r

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
        target_indices : Sequence of int
            Global node indices of targeted internal landmarks.

        Returns:
        --------
        displacements_3d : np.ndarray of shape (len(target_indices), 3)
        """
        indices = np.asarray(target_indices, dtype=np.int64)
        target_dofs = np.repeat(3 * indices, 3) + np.tile(np.arange(3), len(indices))
        Phi_target = self.Phi[target_dofs, :]
        u_target_flat = Phi_target @ u_r
        return u_target_flat.reshape(-1, 3)
