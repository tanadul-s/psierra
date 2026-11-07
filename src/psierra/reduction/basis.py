"""
Proper Orthogonal Decomposition (POD) Reduced Basis and Surface Sub-Basis.

Provides low-dimensional orthonormal coordinate projections:
    u_r = Phi^T u       (project)
    u   approx Phi u_r  (reconstruct)

and pre-sliced surface sub-basis for real-time 3D rendering:
    u_surf = Phi_surf u_r  (reconstruct_surface)
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
import scipy.sparse as spsp


@dataclass(frozen=True)
class SurfacePODBasis:
    """
    Pre-extracted surface sub-basis for instantaneous 3D surface mesh rendering.

    Encapsulates the row-sliced modal projection matrix:
        Phi_surf = Phi[surface_dofs, :] in R^(n_surface_dofs x r)

    Evaluating surface deformations eliminates the O(N) lifting bottleneck during
    online interactive visualization:
        u_surf = Phi_surf @ u_r in R^(3 * N_surf_nodes)
    """

    surface_dofs: np.ndarray
    basis: np.ndarray

    def __post_init__(self) -> None:
        if self.basis.ndim != 2:
            raise ValueError(
                f"Surface basis matrix must be 2D of shape (n_surface_dofs, r), got {self.basis.shape}"
            )
        if len(self.surface_dofs) != self.basis.shape[0]:
            raise ValueError(
                f"Length of surface_dofs ({len(self.surface_dofs)}) does not match "
                f"basis rows ({self.basis.shape[0]})."
            )

    @property
    def n_surface_dofs(self) -> int:
        """Total number of surface degrees of freedom (3 * n_surface_nodes)."""
        return self.basis.shape[0]

    @property
    def n_surface_nodes(self) -> int:
        """Number of surface mesh nodes."""
        return self.basis.shape[0] // 3

    @property
    def n_modes(self) -> int:
        """Number of retained POD modes r."""
        return self.basis.shape[1]

    def reconstruct(self, u_r: np.ndarray) -> np.ndarray:
        """
        Reconstructs the 1D surface displacement vector:
            u_surf = Phi_surf @ u_r in R^(n_surface_dofs).

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
            Generalized reduced displacement coordinates.

        Returns:
        --------
        u_surf : np.ndarray of shape (n_surface_dofs,)
        """
        return self.basis @ u_r

    def reconstruct_surface_displacements(self, u_r: np.ndarray) -> np.ndarray:
        """
        Reconstructs 3D surface nodal displacement vectors formatted for direct
        VTK / PyVista point array mutation (polydata.points = initial + displacements):
            U_surf in R^(N_surf_nodes, 3).

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
            Generalized reduced displacement coordinates.

        Returns:
        --------
        displacements_3d : np.ndarray of shape (N_surf_nodes, 3)
        """
        u_surf_1d = self.basis @ u_r
        return u_surf_1d.reshape(-1, 3)


@dataclass(frozen=True)
class PODBasis:
    """
    Orthonormal Proper Orthogonal Decomposition (POD) basis matrix.

    Contains the spatial principal modes extracted from SVD:
        Phi in R^(n_dofs x r)  with  Phi^T Phi = I_r

    Governing Mathematical Formulas:
    --------------------------------
    Forward projection (Full-to-Reduced):
        u_r = Phi^T @ u in R^r

    Inverse reconstruction (Reduced-to-Full):
        u approx Phi @ u_r in R^n_dofs

    Operator Galerkin projection:
        A_r = Phi^T @ A @ Phi in R^(r x r)
        f_r = Phi^T @ f in R^r
    """

    basis: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: float
    total_energy: float

    def __post_init__(self) -> None:
        if self.basis.ndim != 2:
            raise ValueError(
                f"POD basis matrix must be 2D of shape (n_dofs, r), got {self.basis.shape}"
            )
        if self.singular_values.ndim != 1:
            raise ValueError(
                f"Singular values must be a 1D vector of length r, got {self.singular_values.shape}"
            )
        if self.basis.shape[1] != len(self.singular_values):
            raise ValueError(
                f"Basis columns ({self.basis.shape[1]}) must match singular value count "
                f"({len(self.singular_values)})."
            )

    @property
    def n_dofs(self) -> int:
        """Dimension of the full-order physical model n."""
        return self.basis.shape[0]

    @property
    def n_modes(self) -> int:
        """Number of retained reduced basis modes r."""
        return self.basis.shape[1]

    @property
    def basis_matrix(self) -> np.ndarray:
        """Modal basis projection matrix Phi in R^(n_dofs x r)."""
        return self.basis

    def project(self, u: np.ndarray) -> np.ndarray:
        """
        Projects full-field state vector into reduced coordinates:
            u_r = Phi^T @ u

        Parameters:
        -----------
        u : np.ndarray of shape (n_dofs,)
            Full-field displacement or state vector.

        Returns:
        --------
        u_r : np.ndarray of shape (r,)
            Reduced modal coordinates.
        """
        u_arr = np.asarray(u, dtype=np.float64).ravel()
        if u_arr.size != self.n_dofs:
            raise ValueError(
                f"Cannot project vector of size {u_arr.size} onto basis with {self.n_dofs} DOFs."
            )
        return self.basis.T @ u_arr

    def reconstruct(self, u_r: np.ndarray) -> np.ndarray:
        """
        Reconstructs full-field state from reduced coordinates:
            u approx Phi @ u_r

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
            Reduced modal coordinates.

        Returns:
        --------
        u : np.ndarray of shape (n_dofs,)
            Reconstructed full-field displacement vector.
        """
        u_r_arr = np.asarray(u_r, dtype=np.float64).ravel()
        if u_r_arr.size != self.n_modes:
            raise ValueError(
                f"Cannot reconstruct state from {u_r_arr.size} modes on basis with {self.n_modes} modes."
            )
        return self.basis @ u_r_arr

    def project_matrix(self, A: spsp.spmatrix | np.ndarray) -> np.ndarray:
        """
        Projects a global matrix operator onto the reduced subspace:
            A_r = Phi^T @ A @ Phi in R^(r x r)

        Preserves exact numerical symmetry if A is symmetric.

        Parameters:
        -----------
        A : scipy.sparse matrix or np.ndarray of shape (n_dofs, n_dofs)
            Full-order system matrix (e.g., stiffness K or mass M).

        Returns:
        --------
        A_r : np.ndarray of shape (r, r)
            Dense projected reduced matrix.
        """
        if spsp.issparse(A):
            A_r = self.basis.T @ (A @ self.basis)
        else:
            A_r = self.basis.T @ A @ self.basis

        # Enforce exact symmetry
        A_r = 0.5 * (A_r + A_r.T)
        return A_r

    def project_vector(self, f: np.ndarray) -> np.ndarray:
        """
        Projects a global force vector onto the reduced subspace:
            f_r = Phi^T @ f in R^r

        Parameters:
        -----------
        f : np.ndarray of shape (n_dofs,)
            Full-order external or internal force vector.

        Returns:
        --------
        f_r : np.ndarray of shape (r,)
            Reduced generalized force vector.
        """
        return self.project(f)

    def orthonormality_error(self) -> float:
        """
        Computes the Frobenius norm error of the basis orthonormality:
            ||Phi^T @ Phi - I_r||_F

        Returns:
        --------
        error : float
            Should be near machine precision (< 1e-12).
        """
        I_r = np.eye(self.n_modes, dtype=np.float64)
        return float(np.linalg.norm(self.basis.T @ self.basis - I_r, ord="fro"))

    def extract_surface_basis(
        self,
        surface_nodes_or_dofs: Sequence[int] | np.ndarray,
        is_node_indices: bool = True,
    ) -> SurfacePODBasis:
        """
        Extracts a pre-sliced surface sub-basis for instantaneous 3D surface rendering:
            Phi_surf = Phi[surface_dofs, :] in R^(n_surf_dofs x r)

        Parameters:
        -----------
        surface_nodes_or_dofs : Sequence of int or np.ndarray
            Either 0-based node indices on the surface mesh (if is_node_indices=True)
            or explicit degree-of-freedom indices (if is_node_indices=False).
        is_node_indices : bool, default=True
            If True, expands each node index i to its 3 translational DOFs [3i, 3i+1, 3i+2].

        Returns:
        --------
        surface_basis : SurfacePODBasis
        """
        if is_node_indices:
            nodes = np.asarray(surface_nodes_or_dofs, dtype=np.int64).ravel()
            # Vectorized 3-DOF expansion
            surface_dofs = (3 * nodes[:, None] + np.array([0, 1, 2])).ravel()
        else:
            surface_dofs = np.asarray(surface_nodes_or_dofs, dtype=np.int64).ravel()

        if np.any(surface_dofs < 0) or np.any(surface_dofs >= self.n_dofs):
            raise IndexError(
                f"Surface DOF indices out of range [0, {self.n_dofs - 1}]."
            )

        sliced_matrix = self.basis[surface_dofs, :].copy()
        return SurfacePODBasis(surface_dofs=surface_dofs, basis=sliced_matrix)
