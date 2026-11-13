"""
Hyper-Reduced Nonlinear Mechanical Model (ECSW).

Binds POD modal basis with Energy-Conserving Sampling and Weighting (ECSW)
Reduced Integration Domain to evaluate nonlinear internal forces f_r_int(u_r)
and tangent stiffness K_r_t(u_r) in microseconds without iterating over the full mesh.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from psierra.domain.base import Domain
from psierra.boundary.dirichlet import DirichletManager, DirichletConstraint
from psierra.damping.rayleigh import RayleighDamping
from psierra.reduction.basis import PODBasis, SurfacePODBasis
from psierra.hyperreduction.ecsw import ReducedIntegrationDomain


@dataclass
class _CachedElementECSW:
    """Internal cache entry for a single retained element in the reduced integration domain."""

    elem_idx: int
    weight: float
    nodes: np.ndarray  # (n_elem_nodes,)
    coords: np.ndarray  # (n_elem_nodes, 3) reference coordinates
    Phi_e: np.ndarray  # (n_elem_dofs, r) element modal sub-basis
    elem_form: Any = None
    dofs_per_node: int = 3
    is_shell: bool = False


class HyperReducedECSWModel:
    """
    Hyper-reduced nonlinear continuum model using ECSW empirical cubature.

    Evaluates the projected nonlinear equations of motion in generalized coordinates u_r in R^r:
        M_r * a_r + C_r * v_r + f_r_int(u_r) = f_r(t)

    Key Features:
    -------------
    - Pre-caches element modal sub-bases Phi_e for all p elements in the Reduced Integration Domain (RID).
    - Eliminates all full-order mesh loops during online evaluations (O(p) complexity instead of O(N_e)).
    - Strictly preserves tangent stiffness symmetry: K_r_t = K_r_t^T (energy conservation).
    - Directly reconstructs 3D surface mesh geometry in microseconds via pre-sliced SurfacePODBasis.
    """

    def __init__(
        self,
        domain: Domain,
        basis: PODBasis,
        rid: ReducedIntegrationDomain,
        dirichlet: DirichletManager | Sequence[DirichletConstraint] | None = None,
        damping: RayleighDamping | None = None,
        f_ext_base: np.ndarray | None = None,
        surface_nodes: np.ndarray | None = None,
    ) -> None:
        """
        Parameters:
        -----------
        domain : Domain
            Physical continuum domain (SolidDomain).
        basis : PODBasis
            Retained orthonormal modal basis Phi in R^{n x r}.
        rid : ReducedIntegrationDomain
            ECSW cubature elements and weights (tilde_E, w).
        dirichlet : DirichletManager or list of DirichletConstraint, optional
            Boundary constraint manager.
        damping : RayleighDamping, optional
            Rayleigh damping model.
        f_ext_base : np.ndarray of shape (n_dofs,), optional
            Constant base external force vector (e.g. gravitational body load).
        surface_nodes : np.ndarray, optional
            Node indices belonging to the outer boundary surface for instant rendering.
        """
        self.domain = domain
        self.basis = basis
        self.rid = rid
        self.r = basis.n_modes

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

        # 2. Extract modal basis for free DOFs
        self.Phi_f = basis.basis_matrix[self.free_dofs, :]  # (n_free, r)

        # 3. Project Consistent Mass Matrix: M_r = Phi_f^T * M_ff * Phi_f (in R^{r x r})
        M_global = domain.assemble_mass_matrix()
        M_ff = M_global[self.free_dofs, :][:, self.free_dofs]
        self.M_r = (self.Phi_f.T @ (M_ff @ self.Phi_f)).astype(np.float64)
        self.M_r = 0.5 * (self.M_r + self.M_r.T)

        # 4. Rayleigh Damping Matrix in Reduced Subspace: C_r = alpha * M_r + beta * K_r_0
        self.damping = damping if damping is not None else RayleighDamping(0.0, 0.0)
        K_lin_global = domain.assemble_stiffness_matrix()
        K_lin_ff = K_lin_global[self.free_dofs, :][:, self.free_dofs]
        self.K_r_0 = (self.Phi_f.T @ (K_lin_ff @ self.Phi_f)).astype(np.float64)
        self.K_r_0 = 0.5 * (self.K_r_0 + self.K_r_0.T)
        self.C_r = self.damping.alpha * self.M_r + self.damping.beta * self.K_r_0
        self.C_r = 0.5 * (self.C_r + self.C_r.T)

        # 5. Base External Load in Reduced Coordinates: f_r_base = Phi_f^T * (f_ext_f - K_lin_fp * u_p)
        if f_ext_base is not None:
            f_ext_arr = np.asarray(f_ext_base, dtype=np.float64).ravel()
            f_ext_f = f_ext_arr[self.free_dofs]
            if len(self.prescribed_dofs) > 0 and np.any(self.prescribed_values != 0.0):
                K_fp = K_lin_global[self.free_dofs, :][:, self.prescribed_dofs]
                f_ext_f = f_ext_f - K_fp @ self.prescribed_values
            self.f_r_base = (self.Phi_f.T @ f_ext_f).astype(np.float64)
        else:
            self.f_r_base = np.zeros(self.r, dtype=np.float64)

        # 6. Pre-cache Element Modal Sub-Bases for the Reduced Integration Domain (RID)
        self._cached_elements: list[_CachedElementECSW] = []
        is_coupled = hasattr(domain, "solid_domain") and hasattr(domain, "shell_domain")
        Phi_full = basis.basis_matrix

        if is_coupled:
            n_solid_elems = domain.solid_domain.mesh.n_elements
            solid_nodes = domain.solid_domain.mesh.nodes
            solid_elems = domain.solid_domain.mesh.elements
            solid_form = domain.solid_domain.element

            shell_elems = domain.shell_domain.mesh.elements
            shell_form = domain.shell_domain.element
            shell_coupled_dofs = domain._shell_coupled_dofs
            shell_node_indices = domain.shell_domain.mesh.node_indices

            for elem_idx, weight in zip(self.rid.element_indices, self.rid.weights):
                e_int = int(elem_idx)
                if e_int < n_solid_elems:
                    enodes = solid_elems[e_int]
                    coords = solid_nodes[enodes]
                    edofs = np.concatenate(
                        [np.arange(3 * nid, 3 * nid + 3) for nid in enodes]
                    )
                    Phi_e = Phi_full[edofs, :]
                    self._cached_elements.append(
                        _CachedElementECSW(
                            elem_idx=e_int,
                            weight=float(weight),
                            nodes=enodes,
                            coords=coords,
                            Phi_e=Phi_e,
                            elem_form=solid_form,
                            dofs_per_node=3,
                            is_shell=False,
                        )
                    )
                else:
                    s_idx = e_int - n_solid_elems
                    s_conn = shell_elems[s_idx]
                    g_conn = shell_node_indices[s_conn]
                    coords = solid_nodes[g_conn]
                    edofs = shell_coupled_dofs[s_idx]
                    Phi_e = Phi_full[edofs, :]
                    self._cached_elements.append(
                        _CachedElementECSW(
                            elem_idx=e_int,
                            weight=float(weight),
                            nodes=g_conn,
                            coords=coords,
                            Phi_e=Phi_e,
                            elem_form=shell_form,
                            dofs_per_node=6,
                            is_shell=True,
                        )
                    )
        else:
            elem_form = domain.element
            dofs_per_node = elem_form.dofs_per_node
            elements = domain.mesh.elements
            nodes = domain.mesh.nodes

            for elem_idx, weight in zip(self.rid.element_indices, self.rid.weights):
                elem_nodes = elements[elem_idx]
                coords = nodes[elem_nodes]
                edofs = np.concatenate(
                    [
                        np.arange(dofs_per_node * nid, dofs_per_node * nid + dofs_per_node)
                        for nid in elem_nodes
                    ]
                )
                Phi_e = Phi_full[edofs, :]

                self._cached_elements.append(
                    _CachedElementECSW(
                        elem_idx=int(elem_idx),
                        weight=float(weight),
                        nodes=elem_nodes,
                        coords=coords,
                        Phi_e=Phi_e,
                        elem_form=elem_form,
                        dofs_per_node=dofs_per_node,
                        is_shell=False,
                    )
                )

        # 7. Surface Sub-Basis for Instant Boundary Reconstruction
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

    @property
    def n_modes(self) -> int:
        """Dimension of the reduced subspace (r)."""
        return self.r

    @property
    def n_elements_reduced(self) -> int:
        """Number of retained elements in the reduced integration domain."""
        return self.rid.n_selected

    @property
    def compression_ratio(self) -> float:
        """Fraction of total mesh elements retained in the reduced integration domain."""
        return self.rid.compression_ratio

    @property
    def speedup_factor(self) -> float:
        """Theoretical speedup based on reduced element evaluation count."""
        return self.rid.speedup_factor

    def evaluate_forces_and_stiffness(
        self, u_r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the hyper-reduced nonlinear internal force vector and symmetric
        tangent stiffness matrix using ONLY the elements in the Reduced Integration Domain:

            f_r_int(u_r) = sum_{e in RID} w_e * Phi_e^T * f_int,e(Phi_e * u_r)
            K_r_t(u_r)   = sum_{e in RID} w_e * Phi_e^T * K_t,e(Phi_e * u_r) * Phi_e

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
            Current displacement in reduced generalized coordinates.

        Returns:
        --------
        f_r_int : np.ndarray of shape (r,)
            Hyper-reduced internal force vector.
        K_r_t : np.ndarray of shape (r, r)
            Hyper-reduced symmetric tangent stiffness matrix.
        """
        u_r_vec = np.asarray(u_r, dtype=np.float64).ravel()
        if u_r_vec.size != self.r:
            raise ValueError(f"u_r size ({u_r_vec.size}) must match reduced dimension ({self.r}).")

        f_r_int = np.zeros(self.r, dtype=np.float64)
        K_r_t = np.zeros((self.r, self.r), dtype=np.float64)
        default_form = getattr(self.domain, "element", None)

        for entry in self._cached_elements:
            w = entry.weight
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

            # Accumulate cubature weighted contributions:
            # f_r_int += w * Phi_e^T * f_int_e
            f_r_int += w * (Phi_e.T @ f_int_e)

            # K_r_t += w * Phi_e^T * k_tan_e * Phi_e
            K_r_t += w * (Phi_e.T @ (k_tan_e @ Phi_e))

        # Enforce exact floating-point symmetry: K_r_t = 0.5 * (K_r_t + K_r_t^T)
        K_r_t = 0.5 * (K_r_t + K_r_t.T)
        return f_r_int, K_r_t

    def evaluate_residual(
        self, u_r: np.ndarray, f_ext_r: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the reduced static equilibrium residual vector and tangent stiffness:
            r_r(u_r) = f_r_int(u_r) - f_ext_r

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)
        f_ext_r : np.ndarray of shape (r,), optional

        Returns:
        --------
        residual : np.ndarray of shape (r,)
        K_r_t : np.ndarray of shape (r, r)
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
        """
        Reconstructs the full-order 3D volumetric displacement field u in R^n_dofs.

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)

        Returns:
        --------
        u_full : np.ndarray of shape (n_dofs,)
        """
        u_r_vec = np.asarray(u_r, dtype=np.float64).ravel()
        u_full = np.zeros(self.domain.n_dofs, dtype=np.float64)

        if len(self.prescribed_dofs) > 0:
            u_full[self.prescribed_dofs] = self.prescribed_values

        u_full[self.free_dofs] = self.Phi_f @ u_r_vec
        return u_full

    def reconstruct_surface_displacement(self, u_r: np.ndarray) -> np.ndarray:
        """
        Reconstructs 3D boundary surface displacement vectors directly in microseconds:
            U_surf in R^{N_surf x 3}

        Parameters:
        -----------
        u_r : np.ndarray of shape (r,)

        Returns:
        --------
        U_surf : np.ndarray of shape (N_surf, 3)
        """
        if self.surface_basis is None:
            raise RuntimeError("Surface basis is not initialized. Provide surface_nodes at construction.")
        return self.surface_basis.reconstruct_surface_displacements(u_r)

    def project_load(self, f_ext: np.ndarray) -> np.ndarray:
        """Projects a full-order external load vector onto the reduced subspace."""
        f_arr = np.asarray(f_ext, dtype=np.float64).ravel()
        return self.Phi_f.T @ f_arr[self.free_dofs]

    def set_time_step(self, dt: float) -> None:
        """Sets the active time increment for rate-dependent materials in the model."""
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
