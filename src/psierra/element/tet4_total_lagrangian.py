from __future__ import annotations
import numpy as np

from psierra.element.tet4_base import Tet4Base
from psierra.material.base import Material
from psierra.utility.tensor import tensor_to_voigt_stress


class Tet4TotalLagrangian(Tet4Base):
    """
    Total Lagrangian 4-Node Tetrahedral Solid Element.

    Formulated in the initial undeformed reference configuration X using the
    Green-Lagrange strain tensor E and Second Piola-Kirchhoff (2nd PK) stress tensor S.
    Fully captures large displacements, rotations, and finite deformations.
    """

    def __init__(self, material: Material) -> None:
        super().__init__(material)

    def compute_nonlinear_B_matrix(
        self, H: np.ndarray, u_nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the Total Lagrangian strain-displacement matrix B(u) = B_L + B_NL(u).

        Parameters:
        -----------
        H : np.ndarray of shape (4, 3)
            Shape function gradient matrix.
        u_nodes : np.ndarray of shape (4, 3)
            Current nodal displacements for the 4 vertices.

        Returns:
        --------
        B : np.ndarray of shape (6, 12)
        """
        BL = self.compute_linear_B_matrix(H)
        BNL = np.zeros((6, 12), dtype=np.float64)

        # Displacement gradient l = u^T @ H (shape: 3 x 3)
        # l[i, j] = du_i / dX_j
        l = u_nodes.T @ H
        du, dv, dw = l[0], l[1], l[2]

        for i in range(4):
            b, c, d = H[i, 0], H[i, 1], H[i, 2]
            col = i * 3

            # Normal strain derivatives: du_k/dX_i * dN_a/dX_i
            BNL[0, col : col + 3] = du[0] * b, dv[0] * b, dw[0] * b
            BNL[1, col : col + 3] = du[1] * c, dv[1] * c, dw[1] * c
            BNL[2, col : col + 3] = du[2] * d, dv[2] * d, dw[2] * d

            # Shear strain derivatives
            BNL[3, col : col + 3] = (
                du[0] * c + du[1] * b,
                dv[0] * c + dv[1] * b,
                dw[0] * c + dw[1] * b,
            )
            BNL[4, col : col + 3] = (
                du[1] * d + du[2] * c,
                dv[1] * d + dv[2] * c,
                dw[1] * d + dw[2] * c,
            )
            BNL[5, col : col + 3] = (
                du[0] * d + du[2] * b,
                dv[0] * d + dv[2] * b,
                dw[0] * d + dw[2] * b,
            )

        return BL + BNL

    def compute_nonlinear_force_and_tangent_stiffness(
        self, node_coords: np.ndarray, u_nodes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Total Lagrangian tangent stiffness matrix k_tan (12x12)
        and internal load vector f_int (12,) for the current deformed state.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (4, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (4, 3)
            Nodal displacements u.

        Returns:
        --------
        k_tan : np.ndarray of shape (12, 12)
            Tangent stiffness matrix: k_mat + k_geo.
        f_int : np.ndarray of shape (12,)
            Internal force vector.
        """
        V, H = self.compute_volume_and_gradients(node_coords)

        # Current coordinates: x = X + u
        x = node_coords + u_nodes

        # Deformation gradient: F = x^T @ H (shape: 3 x 3)
        F = x.T @ H
        J = np.linalg.det(F)
        if J <= 0.0:
            raise ValueError(f"Element inversion detected: det(F) = {J:.4e} <= 0.")

        # Query material for constitutive tangent C and 2nd PK stress S
        C, S = self._material.compute_stress_and_tangent(F)

        # Matrices B(u) and G
        B = self.compute_nonlinear_B_matrix(H, u_nodes)
        G = self.compute_geometric_matrix(H)

        # 1. Material Stiffness: k_mat = B^T @ C @ B * V
        k_mat = (B.T @ C @ B) * V

        # 2. Geometric Stiffness: k_geo = G^T @ (I3 ⊗ S) @ G * V
        S_hat = np.kron(np.identity(3, dtype=np.float64), S)
        k_geo = (G.T @ S_hat @ G) * V

        k_tan = k_mat + k_geo

        # 3. Internal Force: f_int = B^T @ S_voigt * V
        S_voigt = tensor_to_voigt_stress(S)
        f_int = (B.T @ S_voigt) * V

        return k_tan, f_int

