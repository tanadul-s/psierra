"""
Discrete Empirical Interpolation Method (DEIM / Q-DEIM) Hyper-Reduction.

Evaluates high-dimensional nonlinear internal forces f_int(u) on a sparse set
of interpolation DOFs and elements without iterating over the full mesh:
1. Force snapshot matrix assembly: S_f = [f_int(u_1), ..., f_int(u_m)] in R^(n x m)
   with support for out-of-core streaming directly into HDF5 datasets.
2. Force subspace extraction via thin SVD: S_f = Psi_f Sigma_f V_f^T.
3. Greedy DEIM (Chaturantabut & Sorensen 2010) or Q-DEIM (Drmac & Gugercin 2016)
   interpolation index selection P in R^(n x m_f).
4. Pre-projected online DEIM operator:
   E_DEIM = Phi_f^T * Psi_f * (P^T * Psi_f)^(-1) in R^(r x m_f)
5. Microsecond online evaluation of hyper-reduced internal forces and tangent stiffness:
   f_r_int(u_r) = E_DEIM * f_sample(u_r)
   K_r_t(u_r)   = E_DEIM * (P^T * K_t * Phi)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Literal, Sequence
import numpy as np
import scipy.linalg as spla
import h5py

from psierra.domain.base import Domain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.damping.rayleigh import RayleighDamping
from psierra.reduction.basis import PODBasis, SurfacePODBasis


def greedy_deim(psi: np.ndarray, m_select: int | None = None) -> np.ndarray:
    """
    Standard Greedy Discrete Empirical Interpolation Method (Chaturantabut & Sorensen, 2010).

    Selects an optimal sequence of m interpolation row indices (DOFs) that minimize
    the interpolation error growth of the basis Psi in R^(n x m):
        p_1 = argmax_i |psi_{i, 0}|
        For l = 1, ..., m - 1:
            solve (P_l^T * U_l) * c = P_l^T * psi_{l+1}
            r = psi_{l+1} - U_l * c
            p_{l+1} = argmax_i |r_i|

    Parameters:
    -----------
    psi : np.ndarray of shape (n, m)
        Orthonormal basis matrix (e.g. from SVD of force snapshots).
    m_select : int, optional
        Number of interpolation indices to select (defaults to m = psi.shape[1]).

    Returns:
    --------
    indices : np.ndarray of shape (m_select,)
        Selected row indices in {0, ..., n - 1}.
    """
    psi_arr = np.asarray(psi, dtype=np.float64)
    if psi_arr.ndim != 2:
        raise ValueError(f"Basis Psi must be 2D, got shape {psi_arr.shape}")
    n, m = psi_arr.shape
    if m == 0 or n == 0:
        raise ValueError(f"Empty basis matrix with shape ({n}, {m})")

    k = m if m_select is None else int(m_select)
    if k <= 0 or k > min(n, m):
        raise ValueError(f"m_select must be in 1..{min(n, m)}, got {k}")

    indices = np.zeros(k, dtype=np.int64)

    # First index: maximum magnitude component of first mode
    indices[0] = int(np.argmax(np.abs(psi_arr[:, 0])))

    if k == 1:
        return indices

    # Iterative greedy selection
    for l in range(1, k):
        # Basis modes selected so far: U_l in R^(n x l)
        U_l = psi_arr[:, :l]
        # Interpolation submatrix: P_l^T * U_l in R^(l x l)
        p_prev = indices[:l]
        P_U = psi_arr[p_prev, :l]

        psi_curr = psi_arr[:, l]
        P_psi = psi_curr[p_prev]

        # Solve interpolation coefficients: P_U * c = P_psi
        try:
            c = spla.solve(P_U, P_psi)
        except Exception:
            c = spla.lstsq(P_U, P_psi)[0]

        # Residual vector
        r = psi_curr - U_l @ c

        # Mask previously selected indices to prevent duplication
        r_abs = np.abs(r)
        r_abs[p_prev] = -1.0

        indices[l] = int(np.argmax(r_abs))

    return indices


def q_deim(psi: np.ndarray, m_select: int | None = None) -> np.ndarray:
    """
    Q-DEIM Interpolation Selection via QR Factorization with Column Pivoting (Drmac & Gugercin, 2016).

    Computes rank-revealing QR factorization on Psi^T in R^(m x n):
        Psi^T * Pi = Q * R
    where Pi is the column permutation matrix selecting the most linearly independent
    columns of Psi^T (corresponding to rows of Psi).

    Parameters:
    -----------
    psi : np.ndarray of shape (n, m)
        Orthonormal basis matrix.
    m_select : int, optional
        Number of interpolation indices to select (defaults to m = psi.shape[1]).

    Returns:
    --------
    indices : np.ndarray of shape (m_select,)
        Selected row indices in {0, ..., n - 1}.
    """
    psi_arr = np.asarray(psi, dtype=np.float64)
    if psi_arr.ndim != 2:
        raise ValueError(f"Basis Psi must be 2D, got shape {psi_arr.shape}")
    n, m = psi_arr.shape
    if m == 0 or n == 0:
        raise ValueError(f"Empty basis matrix with shape ({n}, {m})")

    k = m if m_select is None else int(m_select)
    if k <= 0 or k > min(n, m):
        raise ValueError(f"m_select must be in 1..{min(n, m)}, got {k}")

    # QR with column pivoting on psi^T
    _, _, piv = spla.qr(psi_arr.T, pivoting=True)
    return np.asarray(piv[:k], dtype=np.int64)


@dataclass
class DEIMForceBasis:
    """
    Container for the DEIM force basis and selected interpolation DOFs.

    Attributes:
    -----------
    basis : np.ndarray
        Force subspace basis matrix Psi_f in R^(n_dofs x m_f).
    interpolation_indices : np.ndarray
        Selected interpolation row indices (DOFs) in R^(m_p,).
    singular_values : np.ndarray
        Singular values from thin SVD of the force snapshot matrix.
    cumulative_energy : float
        Fraction of force snapshot energy captured by retained modes (e.g. >= 0.9999).
    total_energy : float
        Sum of squared singular values.
    method : str
        Algorithm used for index selection ('greedy' or 'qdeim').
    metadata : dict
        Optional metadata dictionary (e.g. threshold, mesh name, snapshot count).
    """

    basis: np.ndarray
    interpolation_indices: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: float
    total_energy: float
    method: str = "greedy"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.basis = np.asarray(self.basis, dtype=np.float64)
        self.interpolation_indices = np.asarray(
            self.interpolation_indices, dtype=np.int64
        ).ravel()
        self.singular_values = np.asarray(self.singular_values, dtype=np.float64).ravel()
        self.cumulative_energy = float(self.cumulative_energy)
        self.total_energy = float(self.total_energy)

        if self.basis.ndim != 2:
            raise ValueError(f"Basis matrix must be 2D, got shape {self.basis.shape}")
        if self.interpolation_indices.size == 0:
            raise ValueError("Interpolation indices cannot be empty.")
        if np.any(self.interpolation_indices < 0) or np.any(
            self.interpolation_indices >= self.basis.shape[0]
        ):
            raise ValueError("Interpolation indices out of bounds for basis dimension.")

    @property
    def n_dofs(self) -> int:
        """Physical DOF dimension n."""
        return self.basis.shape[0]

    @property
    def n_modes(self) -> int:
        """Number of retained force modes m_f."""
        return self.basis.shape[1]

    @property
    def n_sampled(self) -> int:
        """Number of selected interpolation points (DOFs) m_p."""
        return int(self.interpolation_indices.size)

    @property
    def submatrix(self) -> np.ndarray:
        """Interpolation submatrix P^T * Psi_f in R^(m_p x m_f)."""
        return self.basis[self.interpolation_indices, :]

    @property
    def condition_number(self) -> float:
        """Condition number of the interpolation submatrix P^T * Psi_f."""
        P_Psi = self.submatrix
        return float(np.linalg.cond(P_Psi))

    def recompute_indices(
        self,
        method: Literal["greedy", "qdeim"] = "greedy",
        m_select: int | None = None,
    ) -> None:
        """Recomputes interpolation indices using the specified algorithm."""
        k = self.n_modes if m_select is None else m_select
        if method == "greedy":
            self.interpolation_indices = greedy_deim(self.basis, m_select=k)
        elif method == "qdeim":
            self.interpolation_indices = q_deim(self.basis, m_select=k)
        else:
            raise ValueError(f"Unknown method '{method}', expected 'greedy' or 'qdeim'")
        self.method = method

    def save(self, path: Path | str, **extra_metadata) -> Path:
        """Saves the DEIM force basis to a compressed .npz archive."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.metadata)
        meta.update(extra_metadata)

        np.savez_compressed(
            p,
            basis=self.basis,
            interpolation_indices=self.interpolation_indices,
            singular_values=self.singular_values,
            cumulative_energy=np.array(self.cumulative_energy),
            total_energy=np.array(self.total_energy),
            method=np.array(self.method),
            metadata=np.array(meta, dtype=object),
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> DEIMForceBasis:
        """Loads a DEIM force basis from a compressed .npz archive."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"DEIM basis archive not found: {p}")

        with np.load(p, allow_pickle=True) as data:
            meta = data["metadata"].item() if "metadata" in data else {}
            method = str(data["method"].item()) if "method" in data else "greedy"
            return cls(
                basis=data["basis"],
                interpolation_indices=data["interpolation_indices"],
                singular_values=data["singular_values"],
                cumulative_energy=float(data["cumulative_energy"]),
                total_energy=float(data["total_energy"]),
                method=method,
                metadata=meta,
            )


class ForceSnapshotManager:
    """
    Collects nonlinear internal force snapshots f_int(u) and computes DEIM force bases.

    Features:
    ---------
    - In-memory collection of force snapshot vectors.
    - Out-of-core streaming directly into an HDF5 dataset to prevent memory exhaustion
      for large 3D solid continuum models.
    - Automated harvesting from displacement snapshot trajectories using Domain constitutive models.
    - Thin SVD with cumulative energy spectrum truncation.
    - Greedy DEIM and Q-DEIM interpolation index selection.
    """

    def __init__(self, n_dofs: int | None = None) -> None:
        """
        Parameters:
        -----------
        n_dofs : int, optional
            Expected degree-of-freedom size of the system.
        """
        self.n_dofs: int | None = int(n_dofs) if n_dofs is not None else None
        self._snapshots: list[np.ndarray] = []
        self._metadata: dict[str, Any] = {}

        # Out-of-core HDF5 streaming state
        self._h5_file: h5py.File | None = None
        self._h5_dataset: h5py.Dataset | None = None
        self._h5_path: Path | None = None

    @property
    def n_snapshots(self) -> int:
        """Total number of collected snapshots."""
        if self._h5_dataset is not None:
            return self._h5_dataset.shape[1]
        return len(self._snapshots)

    @property
    def metadata(self) -> dict[str, Any]:
        """Dictionary of stored simulation metadata."""
        return dict(self._metadata)

    def enable_hdf5_streaming(
        self,
        file_path: Path | str,
        dataset_name: str = "force_snapshots",
        chunk_size: int = 16,
    ) -> None:
        """
        Enables out-of-core streaming of force snapshots directly into an HDF5 dataset.

        Parameters:
        -----------
        file_path : Path or str
            Destination HDF5 file path.
        dataset_name : str, default='force_snapshots'
            Dataset name inside the root group.
        chunk_size : int, default=16
            Number of snapshots per chunk in the HDF5 dataset.
        """
        if self.n_dofs is None:
            raise ValueError(
                "n_dofs must be explicitly specified when initializing ForceSnapshotManager for HDF5 streaming."
            )
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._h5_path = p
        self._h5_file = h5py.File(p, "w")

        # Create resizable dataset: shape (n_dofs, 0), maxshape (n_dofs, None)
        chunk_dofs = min(self.n_dofs, 1024)
        self._h5_dataset = self._h5_file.create_dataset(
            dataset_name,
            shape=(self.n_dofs, 0),
            maxshape=(self.n_dofs, None),
            chunks=(chunk_dofs, max(1, chunk_size)),
            dtype=np.float64,
        )

        # Flush any existing in-memory snapshots to the dataset
        if self._snapshots:
            for s in self._snapshots:
                self._stream_snapshot_to_h5(s)
            self._snapshots.clear()

    def _stream_snapshot_to_h5(self, f_arr: np.ndarray) -> None:
        """Appends a single snapshot column to the active HDF5 dataset."""
        if self._h5_dataset is None:
            return
        curr_m = self._h5_dataset.shape[1]
        self._h5_dataset.resize((self.n_dofs, curr_m + 1))
        self._h5_dataset[:, curr_m] = f_arr

    def close(self) -> None:
        """Flushes and closes active HDF5 file stream."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._h5_dataset = None

    def __enter__(self) -> ForceSnapshotManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def add_snapshot(self, f: np.ndarray) -> None:
        """
        Appends a nonlinear internal force snapshot vector to the collection.

        Parameters:
        -----------
        f : np.ndarray of shape (n_dofs,)
            Internal force vector f_int(u).
        """
        f_arr = np.asarray(f, dtype=np.float64).ravel()
        if self.n_dofs is None:
            self.n_dofs = f_arr.size
        elif f_arr.size != self.n_dofs:
            raise ValueError(
                f"Snapshot size ({f_arr.size}) does not match expected n_dofs ({self.n_dofs})."
            )

        if self._h5_dataset is not None:
            self._stream_snapshot_to_h5(f_arr)
        else:
            self._snapshots.append(f_arr.copy())

    def add_snapshots(self, F: np.ndarray) -> None:
        """Appends multiple force snapshots from a 2D matrix of shape (n_dofs, m)."""
        F_arr = np.asarray(F, dtype=np.float64)
        if F_arr.ndim == 1:
            self.add_snapshot(F_arr)
            return
        if F_arr.ndim != 2:
            raise ValueError(f"Snapshot array must be 1D or 2D, got shape {F_arr.shape}")

        n, m = F_arr.shape
        if self.n_dofs is None:
            self.n_dofs = n
        elif n != self.n_dofs:
            raise ValueError(
                f"Snapshot row count ({n}) does not match expected n_dofs ({self.n_dofs})."
            )

        for col in range(m):
            self.add_snapshot(F_arr[:, col])

    def get_snapshot_matrix(self) -> np.ndarray:
        """
        Returns the full force snapshot matrix S_f in R^(n_dofs x m).

        Reads from in-memory cache or streamed HDF5 file.
        """
        if self._h5_dataset is not None:
            return np.asarray(self._h5_dataset[:], dtype=np.float64)
        if not self._snapshots:
            raise RuntimeError(
                "Cannot construct snapshot matrix: no snapshots have been collected."
            )
        return np.column_stack(self._snapshots)

    def collect_from_displacements(
        self,
        domain: Domain,
        displacements: np.ndarray | Sequence[np.ndarray],
        free_dofs: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Harvests nonlinear internal force snapshots from displacement states.

        For each displacement snapshot u^{(k)}, sets domain.displacement = u^{(k)},
        evaluates f_int(u^{(k)}), and appends it to this manager.

        Parameters:
        -----------
        domain : Domain
            Physical continuum domain (e.g. SolidDomain).
        displacements : np.ndarray of shape (n_dofs, m) or sequence of (n_dofs,) vectors
        free_dofs : np.ndarray, optional
            If specified, stores only the free unconstrained DOFs.

        Returns:
        --------
        S_f : np.ndarray of shape (n_out_dofs, m)
            The collected internal force snapshot matrix.
        """
        if isinstance(displacements, np.ndarray):
            if displacements.ndim == 1:
                disp_mat = displacements.reshape(-1, 1)
            elif displacements.ndim == 2:
                disp_mat = displacements
            else:
                raise ValueError(
                    f"Displacements array must be 1D or 2D, got shape {displacements.shape}"
                )
        elif isinstance(displacements, (list, tuple)):
            if len(displacements) == 0:
                raise ValueError("Displacements sequence cannot be empty.")
            disp_mat = np.column_stack(displacements)
        else:
            raise ValueError(
                f"Displacements must be 2D array or sequence of 1D vectors, got {type(displacements)}"
            )

        n_dofs, m_snapshots = disp_mat.shape
        if n_dofs != domain.n_dofs:
            raise ValueError(
                f"Displacement DOF count ({n_dofs}) does not match domain DOF count ({domain.n_dofs})."
            )

        original_u = domain.u.copy()

        for k in range(m_snapshots):
            u_k = disp_mat[:, k]
            if hasattr(domain, "update_state"):
                domain.update_state(u=u_k)
            else:
                domain.u[:] = u_k
            f_int = domain.assemble_internal_forces()
            if free_dofs is not None:
                f_int = f_int[free_dofs]
            self.add_snapshot(f_int)

        # Restore original displacement state
        if hasattr(domain, "update_state"):
            domain.update_state(u=original_u)
        else:
            domain.u[:] = original_u
        return self.get_snapshot_matrix()

    def compute_basis(
        self,
        energy_threshold: float = 0.9999,
        max_modes: int | None = None,
        method: Literal["greedy", "qdeim"] = "greedy",
        m_sample: int | None = None,
    ) -> DEIMForceBasis:
        """
        Computes thin SVD on the force snapshot matrix and extracts the DEIM basis:
            S_f = Psi_f Sigma_f V_f^T

        Parameters:
        -----------
        energy_threshold : float, default=0.9999
            Cumulative energy retention threshold (0.0 < energy_threshold <= 1.0).
            Default 0.9999 retains 99.99% of nonlinear force energy.
        max_modes : int, optional
            Hard upper limit on the number of retained force modes m_f.
        method : 'greedy' or 'qdeim', default='greedy'
            Algorithm for selecting interpolation DOFs.
        m_sample : int, optional
            Number of interpolation points to select. Defaults to m_f.

        Returns:
        --------
        basis : DEIMForceBasis
        """
        if not (0.0 < energy_threshold <= 1.0):
            raise ValueError(
                f"energy_threshold must be in (0.0, 1.0], got {energy_threshold}"
            )
        if max_modes is not None and max_modes < 1:
            raise ValueError(f"max_modes must be >= 1, got {max_modes}")

        S_f = self.get_snapshot_matrix()

        # Thin SVD: S_f in R^(n x m) -> U in R^(n x m), s in R^m, Vt in R^(m x m)
        U, s, _ = spla.svd(S_f, full_matrices=False)

        energies = s**2
        total_energy = float(np.sum(energies))

        if total_energy < 1e-16:
            r = 1
            cum_energy = 1.0
        else:
            cum_energies = np.cumsum(energies) / total_energy
            r = int(np.searchsorted(cum_energies, energy_threshold) + 1)
            r = min(r, len(s))
            if max_modes is not None:
                r = min(r, max_modes)
            r = max(1, r)
            cum_energy = float(cum_energies[r - 1])

        Psi_f = U[:, :r].copy()

        # Select interpolation indices
        k_sample = r if m_sample is None else int(m_sample)
        if method == "greedy":
            wp = greedy_deim(Psi_f, m_select=k_sample)
        elif method == "qdeim":
            wp = q_deim(Psi_f, m_select=k_sample)
        else:
            raise ValueError(f"Unknown method '{method}', expected 'greedy' or 'qdeim'")

        return DEIMForceBasis(
            basis=Psi_f,
            interpolation_indices=wp,
            singular_values=s[:r].copy(),
            cumulative_energy=cum_energy,
            total_energy=total_energy,
            method=method,
            metadata=dict(self._metadata),
        )


@dataclass
class _CachedElementDEIM:
    """Internal cache entry for an element contributing to the DEIM sampled DOFs."""

    elem_idx: int
    nodes: np.ndarray
    coords: np.ndarray
    Phi_e: np.ndarray  # (n_elem_dofs, r)
    elem_form: Any = None
    dofs_per_node: int = 3
    is_shell: bool = False
    # Scatter mapping from elemental DOFs to sampled DOFs (0..m_p - 1)
    local_dof_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    sample_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


class HyperReducedDEIMModel:
    """
    Hyper-reduced nonlinear continuum model using the Discrete Empirical Interpolation Method (DEIM).

    Evaluates the projected nonlinear equations of motion in generalized coordinates u_r in R^r:
        M_r * a_r + C_r * v_r + f_r_int(u_r) = f_r(t)

    Mathematical Formulation:
    -------------------------
    Nonlinear internal force approximation:
        f_int(u) approx Psi_f * (P^T * Psi_f)^(-1) * f_sample(u)
    where:
        f_sample(u) = P^T * f_int(u) in R^m_p  (only evaluated at interpolation DOFs P)

    Projecting onto the displacement POD modal basis Phi_f:
        f_r_int(u_r) = Phi_f^T * f_int(Phi_f * u_r)
                     approx E_DEIM * f_sample(Phi_f * u_r)
    where the online projection operator:
        E_DEIM = Phi_f^T * Psi_f * (P^T * Psi_f)^(-1) in R^(r x m_p)
    is computed ONCE offline.

    Key Features:
    -------------
    - Pre-caches only the sparse subset of mesh elements connected to interpolation DOFs.
    - Zero full-mesh iterations during online simulation: O(|E_sample|) << O(N_e).
    - Supports both Greedy DEIM and Q-DEIM interpolation.
    - Evaluates reduced tangent stiffness K_r_t(u_r) in microseconds with optional symmetrization.
    - Seamlessly supports rate-dependent viscoelasticity history variables.
    """

    def __init__(
        self,
        domain: Domain,
        basis: PODBasis,
        deim_basis: DEIMForceBasis,
        dirichlet: DirichletManager | Sequence[DirichletConstraint] | None = None,
        damping: RayleighDamping | None = None,
        f_ext_base: np.ndarray | None = None,
        surface_nodes: np.ndarray | None = None,
        symmetrize_stiffness: bool = True,
    ) -> None:
        """
        Parameters:
        -----------
        domain : Domain
            Physical continuum domain (SolidDomain).
        basis : PODBasis
            Retained displacement modal basis Phi in R^(n_dofs x r).
        deim_basis : DEIMForceBasis
            DEIM force basis Psi_f and interpolation row indices P.
        dirichlet : DirichletManager or Sequence[DirichletConstraint], optional
            Boundary constraint manager.
        damping : RayleighDamping, optional
            Rayleigh damping model.
        f_ext_base : np.ndarray of shape (n_dofs,), optional
            Constant base external force vector (e.g. gravitational body load).
        surface_nodes : np.ndarray, optional
            Outer boundary surface nodes for instantaneous visualization.
        symmetrize_stiffness : bool, default=True
            Enforces floating-point symmetry on reduced tangent stiffness K_r_t = 0.5 * (K + K^T).
        """
        self.domain = domain
        self.basis = basis
        self.deim_basis = deim_basis
        self.r = basis.n_modes
        self.symmetrize_stiffness = bool(symmetrize_stiffness)

        # 1. Boundary Condition Partitioning
        if dirichlet is not None:
            if isinstance(dirichlet, list):
                self.dirichlet = DirichletManager(domain.n_dofs)
                for c in dirichlet:
                    self.dirichlet.add_constraint(c)
            else:
                self.dirichlet = dirichlet
            self.free_dofs = self.dirichlet.free_dofs
            self.prescribed_dofs = self.dirichlet.prescribed_dofs
            self.prescribed_values = self.dirichlet.prescribed_values
        else:
            self.dirichlet = None
            self.free_dofs = np.arange(domain.n_dofs, dtype=np.int64)
            self.prescribed_dofs = np.array([], dtype=np.int64)
            self.prescribed_values = np.array([], dtype=np.float64)

        # 2. Extract displacement modal basis for free DOFs
        self.Phi_f = basis.basis_matrix[self.free_dofs, :]  # (n_free, r)

        # 3. Project Consistent Mass Matrix: M_r = Phi_f^T * M_ff * Phi_f
        M_global = domain.assemble_mass_matrix()
        M_ff = M_global[self.free_dofs, :][:, self.free_dofs]
        self.M_r = (self.Phi_f.T @ (M_ff @ self.Phi_f)).astype(np.float64)
        self.M_r = 0.5 * (self.M_r + self.M_r.T)

        # 4. Rayleigh Damping Matrix: C_r = alpha * M_r + beta * K_r_0
        self.damping = damping if damping is not None else RayleighDamping(0.0, 0.0)
        K_lin_global = domain.assemble_stiffness_matrix()
        K_lin_ff = K_lin_global[self.free_dofs, :][:, self.free_dofs]
        self.K_r_0 = (self.Phi_f.T @ (K_lin_ff @ self.Phi_f)).astype(np.float64)
        self.K_r_0 = 0.5 * (self.K_r_0 + self.K_r_0.T)
        self.C_r = self.damping.alpha * self.M_r + self.damping.beta * self.K_r_0
        self.C_r = 0.5 * (self.C_r + self.C_r.T)

        # 5. Base External Load: f_r_base = Phi_f^T * (f_ext_f - K_lin_fp * u_p)
        if f_ext_base is not None:
            f_ext_arr = np.asarray(f_ext_base, dtype=np.float64).ravel()
            f_ext_f = f_ext_arr[self.free_dofs]
            if len(self.prescribed_dofs) > 0 and np.any(self.prescribed_values != 0.0):
                K_fp = K_lin_global[self.free_dofs, :][:, self.prescribed_dofs]
                f_ext_f = f_ext_f - K_fp @ self.prescribed_values
            self.f_r_base = (self.Phi_f.T @ f_ext_f).astype(np.float64)
        else:
            self.f_r_base = np.zeros(self.r, dtype=np.float64)

        # 6. Adapt DEIM basis to free DOFs and compute E_DEIM
        if deim_basis.basis.shape[0] == len(self.free_dofs):
            # Already strictly free DOFs
            self.Psi_f = deim_basis.basis.copy()
            wp_free = deim_basis.interpolation_indices.copy()
            self.sampled_free_dofs = wp_free
            self.sampled_global_dofs = self.free_dofs[wp_free]
        elif deim_basis.basis.shape[0] == domain.n_dofs:
            # Full domain DOFs
            self.Psi_f = deim_basis.basis[self.free_dofs, :].copy()
            # Map indices into free DOFs if possible, or re-select on free DOFs
            raw_wp = deim_basis.interpolation_indices
            in_free = np.isin(raw_wp, self.free_dofs)
            if np.all(in_free):
                sorter = np.argsort(self.free_dofs)
                wp_free = sorter[
                    np.searchsorted(self.free_dofs, raw_wp, sorter=sorter)
                ]
                self.sampled_free_dofs = wp_free
                self.sampled_global_dofs = raw_wp
            else:
                # Re-select interpolation points on free basis
                if deim_basis.method == "qdeim":
                    wp_free = q_deim(self.Psi_f, m_select=deim_basis.n_sampled)
                else:
                    wp_free = greedy_deim(self.Psi_f, m_select=deim_basis.n_sampled)
                self.sampled_free_dofs = wp_free
                self.sampled_global_dofs = self.free_dofs[wp_free]
        else:
            raise ValueError(
                f"DEIM basis row count ({deim_basis.basis.shape[0]}) matches neither domain DOFs "
                f"({domain.n_dofs}) nor free DOFs ({len(self.free_dofs)})."
            )

        self.m_p = len(self.sampled_free_dofs)
        self.m_f = self.Psi_f.shape[1]

        if self.m_f < self.r:
            import warnings

            warnings.warn(
                f"DEIM force basis mode count ({self.m_f}) is less than displacement modal "
                f"dimension ({self.r}). To prevent a rank-deficient reduced tangent stiffness "
                f"matrix during Newton iterations, retain at least m_f >= r force modes.",
                UserWarning,
                stacklevel=2,
            )

        # Interpolation submatrix P^T * Psi_f in R^(m_p x m_f)
        P_Psi = self.Psi_f[self.sampled_free_dofs, :]
        self.P_Psi_cond = float(np.linalg.cond(P_Psi))

        # Online projection matrix: E_DEIM = Phi_f^T * Psi_f * (P^T * Psi_f)^(-1) in R^(r x m_p)
        try:
            inv_P_Psi = spla.pinv(P_Psi)
        except Exception:
            inv_P_Psi = spla.lstsq(P_Psi, np.eye(self.m_p))[0]

        self.E_DEIM = (self.Phi_f.T @ (self.Psi_f @ inv_P_Psi)).astype(np.float64)

        # 7. Identify sampled mesh elements and build fast scatter cache
        self._cached_elements: list[_CachedElementDEIM] = []
        self._build_mesh_cache()

        # 8. Surface sub-basis for rendering
        if surface_nodes is not None:
            self.surface_basis: SurfacePODBasis | None = basis.extract_surface_basis(
                surface_nodes, is_node_indices=True
            )
        elif hasattr(domain, "mesh") and hasattr(domain.mesh, "surface"):
            try:
                surf_node_idx = domain.mesh.surface.node_indices
                if surf_node_idx is not None:
                    self.surface_basis = basis.extract_surface_basis(
                        surf_node_idx, is_node_indices=True
                    )
                else:
                    self.surface_basis = None
            except Exception:
                self.surface_basis = None
        else:
            self.surface_basis = None

    def _build_mesh_cache(self) -> None:
        """Identifies elements touching sampled DOFs and precomputes scatter indices."""
        is_coupled = hasattr(self.domain, "solid_domain") and hasattr(
            self.domain, "shell_domain"
        )
        Phi_full = self.basis.basis_matrix

        # Fast lookup mapping: global DOF -> sample index (0..m_p - 1)
        dof_to_sample: dict[int, int] = {
            int(gdof): idx for idx, gdof in enumerate(self.sampled_global_dofs)
        }
        sampled_nodes_set: set[int] = set()

        if is_coupled:
            solid_mesh = self.domain.solid_domain.mesh
            solid_form = self.domain.solid_domain.element
            solid_elems = solid_mesh.elements
            solid_nodes = solid_mesh.nodes
            n_solid_elems = solid_mesh.n_elements

            shell_mesh = self.domain.shell_domain.mesh
            shell_form = self.domain.shell_domain.element
            shell_elems = shell_mesh.elements
            shell_coupled_dofs = self.domain._shell_coupled_dofs
            shell_node_indices = shell_mesh.node_indices

            for gdof in self.sampled_global_dofs:
                sampled_nodes_set.add(int(gdof // 3))

            # Inspect solid elements
            for e in range(n_solid_elems):
                enodes = solid_elems[e]
                if not any(int(nid) in sampled_nodes_set for nid in enodes):
                    continue

                coords = solid_nodes[enodes]
                edofs = np.concatenate(
                    [np.arange(3 * nid, 3 * nid + 3) for nid in enodes]
                )
                Phi_e = Phi_full[edofs, :]

                # Identify which local DOFs hit sampled DOFs
                local_indices = []
                sample_indices = []
                for loc_idx, gdof in enumerate(edofs):
                    if gdof in dof_to_sample:
                        local_indices.append(loc_idx)
                        sample_indices.append(dof_to_sample[gdof])

                if sample_indices:
                    self._cached_elements.append(
                        _CachedElementDEIM(
                            elem_idx=e,
                            nodes=enodes,
                            coords=coords,
                            Phi_e=Phi_e,
                            elem_form=solid_form,
                            dofs_per_node=3,
                            is_shell=False,
                            local_dof_indices=np.array(local_indices, dtype=np.int64),
                            sample_indices=np.array(sample_indices, dtype=np.int64),
                        )
                    )

            # Inspect shell elements
            for s_idx in range(len(shell_elems)):
                s_conn = shell_elems[s_idx]
                g_conn = shell_node_indices[s_conn]
                edofs = shell_coupled_dofs[s_idx]
                coords = solid_nodes[g_conn]
                Phi_e = Phi_full[edofs, :]

                local_indices = []
                sample_indices = []
                for loc_idx, gdof in enumerate(edofs):
                    if gdof in dof_to_sample:
                        local_indices.append(loc_idx)
                        sample_indices.append(dof_to_sample[gdof])

                if sample_indices:
                    self._cached_elements.append(
                        _CachedElementDEIM(
                            elem_idx=n_solid_elems + s_idx,
                            nodes=g_conn,
                            coords=coords,
                            Phi_e=Phi_e,
                            elem_form=shell_form,
                            dofs_per_node=6,
                            is_shell=True,
                            local_dof_indices=np.array(local_indices, dtype=np.int64),
                            sample_indices=np.array(sample_indices, dtype=np.int64),
                        )
                    )
        else:
            mesh = self.domain.mesh
            elem_form = self.domain.element
            dofs_per_node = elem_form.dofs_per_node
            elements = mesh.elements
            nodes = mesh.nodes

            for gdof in self.sampled_global_dofs:
                sampled_nodes_set.add(int(gdof // dofs_per_node))

            for e in range(mesh.n_elements):
                elem_nodes = elements[e]
                if not any(int(nid) in sampled_nodes_set for nid in elem_nodes):
                    continue

                coords = nodes[elem_nodes]
                edofs = np.concatenate(
                    [
                        np.arange(
                            dofs_per_node * nid,
                            dofs_per_node * nid + dofs_per_node,
                        )
                        for nid in elem_nodes
                    ]
                )
                Phi_e = Phi_full[edofs, :]

                local_indices = []
                sample_indices = []
                for loc_idx, gdof in enumerate(edofs):
                    if gdof in dof_to_sample:
                        local_indices.append(loc_idx)
                        sample_indices.append(dof_to_sample[gdof])

                if sample_indices:
                    self._cached_elements.append(
                        _CachedElementDEIM(
                            elem_idx=e,
                            nodes=elem_nodes,
                            coords=coords,
                            Phi_e=Phi_e,
                            elem_form=elem_form,
                            dofs_per_node=dofs_per_node,
                            is_shell=False,
                            local_dof_indices=np.array(local_indices, dtype=np.int64),
                            sample_indices=np.array(sample_indices, dtype=np.int64),
                        )
                    )

    @property
    def n_modes(self) -> int:
        """Reduced modal dimension r."""
        return self.r

    @property
    def n_sampled_dofs(self) -> int:
        """Number of selected interpolation DOFs m_p."""
        return self.m_p

    @property
    def n_sampled_elements(self) -> int:
        """Number of elements retained in the DEIM cubature sub-mesh."""
        return len(self._cached_elements)

    @property
    def total_elements(self) -> int:
        """Total number of elements in the full physical mesh."""
        if hasattr(self.domain, "solid_domain") and hasattr(
            self.domain, "shell_domain"
        ):
            return (
                self.domain.solid_domain.mesh.n_elements
                + self.domain.shell_domain.mesh.n_elements
            )
        return self.domain.mesh.n_elements

    @property
    def compression_ratio(self) -> float:
        """Ratio of retained elements to total mesh elements."""
        return float(self.n_sampled_elements / max(self.total_elements, 1))

    @property
    def speedup_factor(self) -> float:
        """Theoretical speedup based on element evaluation count."""
        return float(self.total_elements / max(self.n_sampled_elements, 1))

    def evaluate_forces_and_stiffness(
        self, u_r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the hyper-reduced nonlinear internal force vector and tangent
        stiffness matrix using DEIM:

            f_r_int(u_r) = E_DEIM * f_sample(Phi * u_r)
            K_r_t(u_r)   = E_DEIM * (P^T * K_t * Phi)

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
            Current displacement in generalized coordinates.

        Returns:
        --------
        f_r_int : np.ndarray of shape (r,)
        K_r_t : np.ndarray of shape (r, r)
        """
        u_r_vec = np.asarray(u_r, dtype=np.float64).ravel()
        if u_r_vec.size != self.r:
            raise ValueError(
                f"u_r size ({u_r_vec.size}) must match reduced dimension ({self.r})."
            )

        f_sample = np.zeros(self.m_p, dtype=np.float64)
        K_sample_Phi = np.zeros((self.m_p, self.r), dtype=np.float64)
        default_form = getattr(self.domain, "element", None)

        for entry in self._cached_elements:
            Phi_e = entry.Phi_e  # (n_elem_dofs, r)
            coords = entry.coords
            form = entry.elem_form if entry.elem_form is not None else default_form
            dpn = entry.dofs_per_node

            # Local element displacement: u_elem = Phi_e * u_r
            u_elem_flat = Phi_e @ u_r_vec
            u_elem_nodes = u_elem_flat.reshape(-1, dpn)

            # Compute elemental nonlinear tangent stiffness and internal force
            try:
                k_tan_e, f_int_e = form.compute_nonlinear_force_and_tangent_stiffness(
                    coords, u_elem_nodes, elem_index=entry.elem_idx
                )
            except (TypeError, ValueError) as err:
                if isinstance(err, TypeError) or "elem_index" in str(err):
                    k_tan_e, f_int_e = form.compute_nonlinear_force_and_tangent_stiffness(
                        coords, u_elem_nodes
                    )
                else:
                    raise
            f_int_e = f_int_e.ravel()

            loc_idx = entry.local_dof_indices
            samp_idx = entry.sample_indices

            f_sample[samp_idx] += f_int_e[loc_idx]
            K_sample_Phi[samp_idx, :] += k_tan_e[loc_idx, :] @ Phi_e

        # Multiply by precomputed online DEIM operator E_DEIM in R^(r x m_p)
        f_r_int = self.E_DEIM @ f_sample
        K_r_t = self.E_DEIM @ K_sample_Phi

        if self.symmetrize_stiffness:
            K_r_t = 0.5 * (K_r_t + K_r_t.T)

        return f_r_int, K_r_t

    def evaluate_residual(
        self, u_r: np.ndarray, f_ext_r: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the reduced static equilibrium residual vector and tangent stiffness:
            r_r(u_r) = f_r_int(u_r) - f_ext_r
        """
        f_target = (
            np.asarray(f_ext_r, dtype=np.float64).ravel()
            if f_ext_r is not None
            else self.f_r_base
        )
        f_r_int, K_r_t = self.evaluate_forces_and_stiffness(u_r)
        residual = f_r_int - f_target
        return residual, K_r_t

    def reconstruct_displacement(self, u_r: np.ndarray) -> np.ndarray:
        """Reconstructs the full-order 3D volumetric displacement field u in R^n_dofs."""
        u_r_vec = np.asarray(u_r, dtype=np.float64).ravel()
        u_full = np.zeros(self.domain.n_dofs, dtype=np.float64)

        if len(self.prescribed_dofs) > 0:
            u_full[self.prescribed_dofs] = self.prescribed_values

        u_full[self.free_dofs] = self.Phi_f @ u_r_vec
        return u_full

    def reconstruct_surface_displacement(self, u_r: np.ndarray) -> np.ndarray:
        """Reconstructs 3D boundary surface displacement vectors in microseconds."""
        if self.surface_basis is None:
            raise RuntimeError(
                "Surface basis is not initialized. Provide surface_nodes at construction."
            )
        return self.surface_basis.reconstruct_surface_displacements(u_r)

    def project_load(self, f_ext: np.ndarray) -> np.ndarray:
        """Projects a full-order external load vector onto the reduced subspace."""
        f_arr = np.asarray(f_ext, dtype=np.float64).ravel()
        return self.Phi_f.T @ f_arr[self.free_dofs]

    def set_time_step(self, dt: float) -> None:
        """Sets active time increment for rate-dependent materials."""
        if hasattr(self.domain, "set_time_step"):
            self.domain.set_time_step(dt)

    def commit_state(self) -> None:
        """Commits converged equilibrium state for history-dependent materials."""
        if hasattr(self.domain, "commit_state"):
            self.domain.commit_state()

    def revert_state(self) -> None:
        """Reverts trial state buffers for history-dependent materials upon cutback."""
        if hasattr(self.domain, "revert_state"):
            self.domain.revert_state()

    def reset_state(self) -> None:
        """Resets all internal history variables in the physical domain."""
        if hasattr(self.domain, "reset_state"):
            self.domain.reset_state()
