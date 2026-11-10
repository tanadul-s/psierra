from __future__ import annotations
import numpy as np

from psierra.element.tet4_base import Tet4Base
from psierra.material.base import Material
from psierra.utility.tensor import (
    tensor_to_voigt_stress,
    compute_strain_pushforward_matrix,
)


class Tet4UpdatedLagrangian(Tet4Base):
    """
    Updated Lagrangian 4-Node Linear Tetrahedral Solid Element.

    Formulated in the current deformed configuration x = X + u using the
    Cauchy stress tensor sigma, spatial velocity/strain rate gradients h = dN/dx,
    spatial linear strain-displacement matrix b_L, and spatial geometric matrix g.

    Yields mathematically identical virtual work and consistent Newton-Raphson tangent
    stiffness as Total Lagrangian, while evaluating all quantities directly in the
    current spatial geometry.
    """

    supports_element_indexing: bool = True

    def __init__(self, material: Material) -> None:
        super().__init__(material)
        self.supports_element_indexing = True

    def compute_nonlinear_force_and_tangent_stiffness(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
        elem_index: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Updated Lagrangian tangent stiffness matrix k_tan (12x12)
        and internal force vector f_int (12,) in the current deformed configuration.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference nodal coordinates X.
        u_nodes : np.ndarray of shape (4, 3)
            Current nodal displacements u.
        elem_index : int, optional
            Global element index for history-dependent materials (e.g. Viscoelastic).

        Returns:
        --------
        k_tan : np.ndarray of shape (12, 12)
            Spatial tangent stiffness matrix: k_mat + k_geo.
        f_int : np.ndarray of shape (12,)
            Spatial internal force vector: b_L^T @ sigma_voigt * v.
        """
        V, H = self.compute_volume_and_gradients(node_coords)

        # Current deformed coordinates: x = X + u
        x = node_coords + u_nodes

        # Deformation gradient: F = x^T @ H (3x3)
        F = x.T @ H
        J = float(np.linalg.det(F))
        if J <= 0.0:
            raise ValueError(f"Element inversion detected in Tet4UpdatedLagrangian: det(F) = {J:.4e} <= 0.")

        # Spatial shape function gradients: h = H @ F^{-1} (4x3)
        F_inv = np.linalg.inv(F)
        h = H @ F_inv

        # Current deformed element volume: v = J * V
        v = J * V

        # Query material model for constitutive tangent C and 2nd PK stress S
        if elem_index is not None and getattr(
            self._material, "supports_element_indexing", False
        ):
            C, S = self._material.compute_stress_and_tangent(F, elem_index=elem_index)
        else:
            C, S = self._material.compute_stress_and_tangent(F)

        # Cauchy stress tensor: sigma = (1 / J) * F @ S @ F^T
        sigma = (1.0 / J) * (F @ S @ F.T)
        sigma_voigt = tensor_to_voigt_stress(sigma)

        # Spatial linear strain-displacement matrix b_L (6x12)
        b_L = self.compute_linear_B_matrix(h)

        # 1. Internal force vector in current configuration:
        # f_int = int_{v} b_L^T @ sigma_voigt dv = b_L^T @ sigma_voigt * v
        f_int = (b_L.T @ sigma_voigt) * v

        # 2. Spatial material tangent modulus:
        # c_spatial = (1 / J) * T_eps^T @ C @ T_eps
        T_eps = compute_strain_pushforward_matrix(F)
        c_spatial = (1.0 / J) * (T_eps.T @ C @ T_eps)
        k_mat = (b_L.T @ c_spatial @ b_L) * v

        # 3. Spatial geometric stiffness:
        # k_geo = int_{v} g^T @ (I3 (x) sigma) @ g dv = g^T @ (I3 (x) sigma) @ g * v
        g = self.compute_geometric_matrix(h)
        I3 = np.identity(3, dtype=np.float64)
        sigma_hat = np.kron(I3, sigma)
        k_geo = (g.T @ sigma_hat @ g) * v

        k_tan = k_mat + k_geo
        return k_tan, f_int

