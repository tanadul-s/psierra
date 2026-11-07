"""
Reduced-Order Linear Structural Model.

Encapsulates projected low-dimensional operators:
    M_r = Phi_f^T @ M_ff @ Phi_f in R^(r x r)
    K_r = Phi_f^T @ K_ff @ Phi_f in R^(r x r)
    C_r = Phi_f^T @ C_ff @ Phi_f in R^(r x r)
    f_r = Phi_f^T @ (f_ext_f - K_fp @ u_p) in R^r

Provides Dirichlet partitioning, coordinate transformations, and instant
3D surface reconstruction.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
import scipy.sparse as spsp

from psierra.domain.base import Domain
from psierra.boundary.dirichlet import (
    DirichletManager,
    DirichletConstraint,
    DirichletBoundaryCondition,
)
from psierra.damping.rayleigh import RayleighDamping
from psierra.reduction.basis import PODBasis, SurfacePODBasis


@dataclass
class ReducedLinearModel:
    """
    Projected linear structural system model with Dirichlet boundary handling.

    Retains the reduced system matrices (M_r, K_r, C_r) and load vectors (f_r)
    in a compact, dense format for real-time dynamic integration.
    """

    basis: PODBasis
    M_r: np.ndarray
    K_r: np.ndarray
    C_r: np.ndarray
    f_r_base: np.ndarray
    total_dofs: int
    free_dofs: np.ndarray
    prescribed_dofs: np.ndarray
    prescribed_values: np.ndarray
    Phi_f: np.ndarray  # Sliced basis on free DOFs: shape (len(free_dofs), r)
    surface_basis: SurfacePODBasis | None = None

    def __post_init__(self) -> None:
        r = self.basis.n_modes
        for name, mat in [("M_r", self.M_r), ("K_r", self.K_r), ("C_r", self.C_r)]:
            if mat.shape != (r, r):
                raise ValueError(
                    f"Reduced matrix {name} must have shape ({r}, {r}), got {mat.shape}."
                )

    @property
    def n_modes(self) -> int:
        """Number of retained POD modes r."""
        return self.basis.n_modes

    @property
    def n_free_dofs(self) -> int:
        """Number of free degrees of freedom in the full-order model."""
        return len(self.free_dofs)

    def project_displacement(self, u_full: np.ndarray) -> np.ndarray:
        """
        Projects full-order displacement field onto reduced coordinates:
            u_r = Phi_f^T @ (u_full[free_dofs])

        Parameters:
        -----------
        u_full : np.ndarray of shape (total_dofs,)

        Returns:
        --------
        u_r : np.ndarray of shape (r,)
        """
        u_f = u_full[self.free_dofs]
        return self.Phi_f.T @ u_f

    def reconstruct_displacement(self, u_r: np.ndarray) -> np.ndarray:
        """
        Reconstructs full-order displacement field from reduced coordinates:
            u_full[free_dofs] = Phi_f @ u_r
            u_full[prescribed_dofs] = prescribed_values

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)

        Returns:
        --------
        u_full : np.ndarray of shape (total_dofs,)
        """
        u_full = np.zeros(self.total_dofs, dtype=np.float64)
        if len(self.prescribed_dofs) > 0:
            u_full[self.prescribed_dofs] = self.prescribed_values
        u_full[self.free_dofs] = self.Phi_f @ u_r
        return u_full

    def reconstruct_surface_displacement(self, u_r: np.ndarray) -> np.ndarray:
        """
        Reconstructs 3D surface displacements directly:
            U_surf = Phi_surf @ u_r in R^(N_surf_nodes, 3)

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)

        Returns:
        --------
        displacements_3d : np.ndarray of shape (N_surf_nodes, 3)
        """
        if self.surface_basis is None:
            raise RuntimeError(
                "No SurfacePODBasis attached to ReducedLinearModel. "
                "Pass surface_nodes during model creation."
            )
        return self.surface_basis.reconstruct_surface_displacements(u_r)

    def project_load(self, f_ext: np.ndarray) -> np.ndarray:
        """
        Projects an arbitrary full-order external load vector onto reduced coordinates:
            f_r = Phi_f^T @ f_ext[free_dofs]

        Parameters:
        -----------
        f_ext : np.ndarray of shape (total_dofs,)

        Returns:
        --------
        f_r : np.ndarray of shape (r,)
        """
        f_ext_arr = np.asarray(f_ext, dtype=np.float64).ravel()
        if f_ext_arr.size != self.total_dofs:
            raise ValueError(
                f"External force vector size {f_ext_arr.size} does not match "
                f"total DOFs {self.total_dofs}."
            )
        return self.Phi_f.T @ f_ext_arr[self.free_dofs]

    @classmethod
    def from_domain(
        cls,
        domain: Domain,
        basis: PODBasis,
        dirichlet: DirichletManager | Sequence[DirichletConstraint] | None = None,
        damping: RayleighDamping | None = None,
        f_ext_base: np.ndarray | None = None,
        surface_nodes: np.ndarray | Sequence[int] | None = None,
    ) -> ReducedLinearModel:
        """
        Assembles full-order domain matrices, applies boundary partitioning,
        and projects structural operators into the POD subspace.

        Parameters:
        -----------
        domain : Domain
            Physical domain (SolidDomain, ShellDomain, CoupledSolidShellDomain).
        basis : PODBasis
            Computed orthonormal reduced basis.
        dirichlet : DirichletManager or Sequence of DirichletConstraint, optional
            Prescribed essential boundary conditions.
        damping : RayleighDamping, optional
            Rayleigh damping model C = alpha * M + beta * K.
        f_ext_base : np.ndarray of shape (total_dofs,), optional
            Constant base external force vector (e.g. gravitational body force).
        surface_nodes : np.ndarray or Sequence of int, optional
            0-based node indices belonging to the outer visual surface for instant rendering.
            If None and the mesh has a .surface attribute, extracted automatically.

        Returns:
        --------
        model : ReducedLinearModel
        """
        total_dofs = domain.n_dofs

        # 1. Resolve Dirichlet Partitioning
        if isinstance(dirichlet, DirichletManager):
            dir_mgr = dirichlet
        elif isinstance(dirichlet, Sequence):
            dir_mgr = DirichletManager(total_dofs)
            for c in dirichlet:
                if isinstance(c, (DirichletConstraint, DirichletBoundaryCondition)):
                    dir_mgr.add_constraint(c)
        else:
            dir_mgr = DirichletManager(total_dofs)

        free_dofs = (
            dir_mgr.free_dofs()
            if callable(dir_mgr.free_dofs)
            else dir_mgr.free_dofs
        )
        p_dofs = (
            dir_mgr.prescribed_dofs()
            if callable(dir_mgr.prescribed_dofs)
            else dir_mgr.prescribed_dofs
        )
        u_p = (
            dir_mgr.prescribed_values()
            if callable(dir_mgr.prescribed_values)
            else dir_mgr.prescribed_values
        )

        # 2. Assemble Global System Matrices
        K = domain.assemble_stiffness_matrix()
        M = domain.assemble_mass_matrix()

        damp_model = damping if damping is not None else RayleighDamping(0.0, 0.0)
        C = damp_model.assemble_damping_matrix(M, K)

        # 3. Partition Matrices to Free DOFs
        K_ff = K[free_dofs, :][:, free_dofs]
        M_ff = M[free_dofs, :][:, free_dofs]
        C_ff = C[free_dofs, :][:, free_dofs]

        # 4. Resolve Basis Slicing
        # If basis is full DOFs (shape n x r), slice free DOFs
        if basis.n_dofs == total_dofs:
            Phi_f = basis.basis[free_dofs, :].copy()
        elif basis.n_dofs == len(free_dofs):
            Phi_f = basis.basis.copy()
        else:
            raise ValueError(
                f"Basis DOF dimension ({basis.n_dofs}) matches neither total DOFs "
                f"({total_dofs}) nor free DOFs ({len(free_dofs)})."
            )

        # 5. Project Matrices to Reduced Coordinates: A_r = Phi_f^T @ A_ff @ Phi_f
        M_r = Phi_f.T @ (M_ff @ Phi_f)
        K_r = Phi_f.T @ (K_ff @ Phi_f)
        C_r = Phi_f.T @ (C_ff @ Phi_f)

        # Enforce exact symmetry
        M_r = 0.5 * (M_r + M_r.T)
        K_r = 0.5 * (K_r + K_r.T)
        C_r = 0.5 * (C_r + C_r.T)

        # 6. Project Base External Load: f_ext_f - K_fp @ u_p
        f_base = (
            np.asarray(f_ext_base, dtype=np.float64).ravel()
            if f_ext_base is not None
            else np.zeros(total_dofs, dtype=np.float64)
        )
        if len(p_dofs) > 0 and np.any(u_p != 0.0):
            K_fp = K[free_dofs, :][:, p_dofs]
            f_reduced_f = f_base[free_dofs] - K_fp @ u_p
        else:
            f_reduced_f = f_base[free_dofs]

        f_r_base = Phi_f.T @ f_reduced_f

        # 7. Extract Surface Sub-Basis for Real-Time Rendering
        surface_basis: SurfacePODBasis | None = None
        if surface_nodes is not None:
            surface_basis = basis.extract_surface_basis(surface_nodes, is_node_indices=True)
        elif hasattr(domain, "mesh") and hasattr(domain.mesh, "surface"):
            try:
                surf_node_idx = domain.mesh.surface.node_indices
                if surf_node_idx is not None:
                    surface_basis = basis.extract_surface_basis(
                        surf_node_idx, is_node_indices=True
                    )
            except Exception:
                surface_basis = None

        return cls(
            basis=basis,
            M_r=M_r,
            K_r=K_r,
            C_r=C_r,
            f_r_base=f_r_base,
            total_dofs=total_dofs,
            free_dofs=free_dofs,
            prescribed_dofs=p_dofs,
            prescribed_values=u_p,
            Phi_f=Phi_f,
            surface_basis=surface_basis,
        )
