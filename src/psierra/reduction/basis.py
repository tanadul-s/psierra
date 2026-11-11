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
import scipy.linalg as spla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla


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


@dataclass(frozen=True)
class EnrichedPODBasis(PODBasis):
    """
    Contact-Enriched Modal Basis: Phi = [Phi_bulk, Phi_contact] in R^(n_dofs x (r_bulk + r_contact)).

    Combines:
    1. Global low-frequency POD displacement modes Phi_bulk (bending, vibration, bulk compliance).
    2. Localized contact enrichment modes Phi_contact (Krylov contact vectors, contact force response,
       or contact displacement residual modes), orthogonalized against Phi_bulk:
           Phi_contact_ortho = (I - Phi_bulk @ Phi_bulk^T) @ Phi_contact_raw
       followed by QR/SVD orthonormalization.

    Guarantees:
        Phi_enriched^T @ Phi_enriched = I_r to machine precision (< 1e-12).
    """

    n_bulk_modes: int = 0
    n_contact_modes: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_bulk_modes < 0 or self.n_contact_modes < 0:
            raise ValueError("n_bulk_modes and n_contact_modes must be non-negative.")
        if self.n_bulk_modes + self.n_contact_modes != self.n_modes:
            raise ValueError(
                f"Sum of bulk ({self.n_bulk_modes}) and contact ({self.n_contact_modes}) modes "
                f"must equal total modes ({self.n_modes})."
            )

    @property
    def bulk_basis(self) -> np.ndarray:
        """Modal sub-matrix corresponding to bulk deformation modes in R^(n_dofs x r_bulk)."""
        return self.basis[:, : self.n_bulk_modes]

    @property
    def contact_basis(self) -> np.ndarray:
        """Modal sub-matrix corresponding to contact enrichment modes in R^(n_dofs x r_contact)."""
        return self.basis[:, self.n_bulk_modes :]

    @classmethod
    def from_snapshots_and_contact(
        cls,
        bulk_snapshots: np.ndarray,
        contact_snapshots: np.ndarray | None = None,
        contact_forces: np.ndarray | None = None,
        K_0: spsp.spmatrix | np.ndarray | None = None,
        energy_threshold_bulk: float = 0.999,
        max_bulk_modes: int | None = None,
        energy_threshold_contact: float = 0.99,
        max_contact_modes: int | None = None,
    ) -> EnrichedPODBasis:
        """
        Constructs an orthonormal contact-enriched modal basis from bulk snapshots
        and localized contact responses.

        Parameters:
        -----------
        bulk_snapshots : np.ndarray of shape (n_dofs, m_bulk)
            Displacement snapshots capturing free / global deformation.
        contact_snapshots : np.ndarray of shape (n_dofs, m_contact), optional
            Displacement snapshots capturing states with active contact.
        contact_forces : np.ndarray of shape (n_dofs, m_forces), optional
            Nodal contact force snapshots from contact interfaces.
        K_0 : spmatrix or ndarray of shape (n_dofs, n_dofs), optional
            Linear or tangent stiffness matrix used to generate Krylov contact
            compliance modes (K_0^{-1} f_c).
        energy_threshold_bulk : float, default=0.999
            Cumulative energy threshold for bulk POD modes.
        max_bulk_modes : int, optional
            Maximum allowed bulk modes.
        energy_threshold_contact : float, default=0.99
            Cumulative energy threshold for contact enrichment modes.
        max_contact_modes : int, optional
            Maximum allowed contact enrichment modes.

        Returns:
        --------
        basis : EnrichedPODBasis
        """
        S_bulk = np.asarray(bulk_snapshots, dtype=np.float64)
        if S_bulk.ndim == 1:
            S_bulk = S_bulk.reshape(-1, 1)
        n_dofs = S_bulk.shape[0]

        # 1. Bulk POD Basis Extraction
        U_b, s_b, _ = spla.svd(S_bulk, full_matrices=False)
        energies_b = s_b**2
        total_energy_b = float(np.sum(energies_b))

        if total_energy_b < 1e-16:
            r_bulk = 1
            cum_energy_b = 1.0
        else:
            cum_energies_b = np.cumsum(energies_b) / total_energy_b
            r_bulk = int(np.searchsorted(cum_energies_b, energy_threshold_bulk) + 1)
            r_bulk = min(r_bulk, len(s_b))
            if max_bulk_modes is not None:
                r_bulk = min(r_bulk, max_bulk_modes)
            r_bulk = max(1, r_bulk)
            cum_energy_b = float(cum_energies_b[r_bulk - 1])

        Phi_bulk = U_b[:, :r_bulk].copy()

        # 2. Extract Candidate Contact Enrichment Vectors
        contact_vectors: list[np.ndarray] = []

        # 2A. Contact force Krylov / static compliance modes: psi = K_0^{-1} f_c
        if contact_forces is not None and K_0 is not None:
            F_c = np.asarray(contact_forces, dtype=np.float64)
            if F_c.ndim == 1:
                F_c = F_c.reshape(-1, 1)
            for col in range(F_c.shape[1]):
                f_vec = F_c[:, col]
                if np.linalg.norm(f_vec) > 1e-10:
                    try:
                        if spsp.issparse(K_0):
                            psi = spspla.spsolve(K_0.tocsc(), f_vec)
                        else:
                            psi = np.linalg.solve(K_0, f_vec)
                        if np.all(np.isfinite(psi)) and np.linalg.norm(psi) > 1e-12:
                            contact_vectors.append(psi / np.linalg.norm(psi))
                    except Exception:
                        pass

        # 2B. Contact displacement residual modes: r = u_c - Phi_bulk @ (Phi_bulk^T u_c)
        if contact_snapshots is not None:
            S_contact = np.asarray(contact_snapshots, dtype=np.float64)
            if S_contact.ndim == 1:
                S_contact = S_contact.reshape(-1, 1)
            R_contact = S_contact - Phi_bulk @ (Phi_bulk.T @ S_contact)
            r_norms = np.linalg.norm(R_contact, axis=0)
            for col in range(S_contact.shape[1]):
                if r_norms[col] > 1e-8:
                    contact_vectors.append(R_contact[:, col] / r_norms[col])

        # 3. Orthogonalize Contact Modes Against Bulk Subspace
        r_contact = 0
        Q_contact = np.empty((n_dofs, 0), dtype=np.float64)

        if contact_vectors:
            Psi_raw = np.column_stack(contact_vectors)
            # Project out bulk subspace: (I - Phi_bulk @ Phi_bulk^T) @ Psi_raw
            Psi_ortho = Psi_raw - Phi_bulk @ (Phi_bulk.T @ Psi_raw)
            col_norms = np.linalg.norm(Psi_ortho, axis=0)
            valid_mask = col_norms > 1e-8

            if np.any(valid_mask):
                Psi_valid = Psi_ortho[:, valid_mask]
                U_c, s_c, _ = spla.svd(Psi_valid, full_matrices=False)
                energies_c = s_c**2
                total_energy_c = float(np.sum(energies_c))

                if total_energy_c > 1e-16:
                    cum_energies_c = np.cumsum(energies_c) / total_energy_c
                    r_c = int(np.searchsorted(cum_energies_c, energy_threshold_contact) + 1)
                    r_c = min(r_c, len(s_c))
                    if max_contact_modes is not None:
                        r_c = min(r_c, max_contact_modes)
                    r_c = max(1, r_c)

                    # Two-pass Gram-Schmidt / QR for machine-precision orthogonality
                    Phi_cand = U_c[:, :r_c].copy()
                    Phi_cand = Phi_cand - Phi_bulk @ (Phi_bulk.T @ Phi_cand)
                    Q_c, R_qr = np.linalg.qr(Phi_cand, mode="reduced")
                    # Filter rank-deficient columns
                    r_diag = np.abs(np.diag(R_qr))
                    cutoff = 1e-8 * (np.max(r_diag) if r_diag.size > 0 else 1.0)
                    keep = np.where(r_diag > cutoff)[0]
                    Q_contact = Q_c[:, keep]
                    # Second pass
                    Q_contact = Q_contact - Phi_bulk @ (Phi_bulk.T @ Q_contact)
                    Q_contact, _ = np.linalg.qr(Q_contact, mode="reduced")
                    r_contact = Q_contact.shape[1]

        # 4. Concatenate and Package Orthonormal Enriched Basis
        if r_contact > 0:
            Phi_enriched = np.column_stack([Phi_bulk, Q_contact])
            s_enriched = np.concatenate([s_b[:r_bulk], np.ones(r_contact)])
        else:
            Phi_enriched = Phi_bulk
            s_enriched = s_b[:r_bulk]

        return cls(
            basis=Phi_enriched,
            singular_values=s_enriched,
            cumulative_energy=cum_energy_b,
            total_energy=total_energy_b,
            n_bulk_modes=r_bulk,
            n_contact_modes=r_contact,
        )
