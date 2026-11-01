from __future__ import annotations
import numpy as np

from psierra.element.quadrature import get_tetrahedral_quadrature
from psierra.element.tet10_base import Tet10Base
from psierra.material.base import Material
from psierra.utility.tensor import (
    tensor_to_voigt_stress,
    compute_strain_pushforward_matrix,
)


class Tet10UpdatedLagrangian(Tet10Base):
    """
    Updated Lagrangian 10-Node Quadratic Tetrahedral Solid Element.

    Formulated in the current deformed configuration x = X + u using spatial
    shape function gradients h = dN/dx, the spatial linear strain-displacement matrix b_L,
    the Cauchy stress tensor sigma, and the spatial material tangent modulus c_spatial.

    Integrates material and geometric stiffness and internal forces over the deformed volume
    using 4-point Hammer numerical quadrature.
    """

    def __init__(self, material: Material, default_quadrature_order: int = 4) -> None:
        super().__init__(material, default_quadrature_order=default_quadrature_order)

    def compute_nonlinear_force_and_tangent_stiffness(
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
        quadrature_order: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Updated Lagrangian tangent stiffness matrix k_tan (30x30)
        and internal force vector f_int (30,) via numerical quadrature in the current configuration.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (10, 3)
            Reference nodal coordinates X.
        u_nodes : np.ndarray of shape (10, 3)
            Current nodal displacements u.
        quadrature_order : int, optional
            Number of Gauss integration points (default: 4).

        Returns:
        --------
        k_tan : np.ndarray of shape (30, 30)
            Spatial tangent stiffness matrix: k_mat + k_geo.
        f_int : np.ndarray of shape (30,)
            Spatial internal force vector.
        """
        order = (
            self._default_quadrature_order
            if quadrature_order is None
            else quadrature_order
        )
        qp, qw = get_tetrahedral_quadrature(order)

        x = node_coords + u_nodes
        k_tan = np.zeros((30, 30), dtype=np.float64)
        f_int = np.zeros(30, dtype=np.float64)
        I3 = np.identity(3, dtype=np.float64)

        for q in range(len(qw)):
            dN_dxi = self.evaluate_shape_gradients_ref(*qp[q])

            # Reference Jacobian and gradients
            J_ref = node_coords.T @ dN_dxi
            detJ_ref = np.linalg.det(J_ref)
            if detJ_ref <= 0.0:
                raise ValueError(
                    f"Negative or zero reference Jacobian determinant: det(J) = {detJ_ref:.4e}."
                )
            H = dN_dxi @ np.linalg.inv(J_ref)

            # Deformation gradient F
            F = x.T @ H
            detF = float(np.linalg.det(F))
            if detF <= 0.0:
                raise ValueError(
                    f"Element inversion detected in Tet10UpdatedLagrangian: det(F) = {detF:.4e} <= 0."
                )

            # Current spatial Jacobian j = x^T @ dN_dxi and spatial gradients h = dN/dx
            J_curr = x.T @ dN_dxi
            detJ_curr = np.linalg.det(J_curr)
            if detJ_curr <= 0.0:
                raise ValueError(
                    f"Negative or zero current Jacobian determinant: det(j) = {detJ_curr:.4e}."
                )
            h = dN_dxi @ np.linalg.inv(J_curr)

            # Spatial strain-displacement and geometric matrices
            b_L = self.compute_linear_B_matrix(h)
            g = self.compute_geometric_matrix(h)

            # Material response
            C, S = self._material.compute_stress_and_tangent(F)

            # Cauchy stress: sigma = (1 / detF) * F @ S @ F^T
            sigma = (1.0 / detF) * (F @ S @ F.T)
            sigma_voigt = tensor_to_voigt_stress(sigma)

            # Spatial material tangent modulus: c_spatial = (1 / detF) * T_eps^T @ C @ T_eps
            T_eps = compute_strain_pushforward_matrix(F)
            c_spatial = (1.0 / detF) * (T_eps.T @ C @ T_eps)

            # Quadrature differential volume: dv = qw[q] * det(j)
            dv = qw[q] * detJ_curr

            # Internal force contribution: f_int += b_L^T @ sigma_voigt * dv
            f_int += (b_L.T @ sigma_voigt) * dv

            # Material and geometric tangent contributions
            k_mat = b_L.T @ c_spatial @ b_L
            sigma_hat = np.kron(I3, sigma)
            k_geo = g.T @ sigma_hat @ g

            k_tan += (k_mat + k_geo) * dv

        return k_tan, f_int

