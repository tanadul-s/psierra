from __future__ import annotations
import numpy as np

from psierra.element.quadrature import get_tetrahedral_quadrature
from psierra.element.tet10_base import Tet10Base
from psierra.material.base import Material
from psierra.utility.tensor import tensor_to_voigt_stress


class Tet10TotalLagrangian(Tet10Base):
    """
    Total Lagrangian 10-Node Quadratic Tetrahedral Solid Element.

    Formulated in the initial undeformed reference configuration X using the
    Green-Lagrange strain tensor E and Second Piola-Kirchhoff (2nd PK) stress tensor S.
    Features 4-point Hammer numerical quadrature integration.
    """

    def __init__(self, material: Material, default_quadrature_order: int = 4) -> None:
        super().__init__(material, default_quadrature_order=default_quadrature_order)

    def compute_nonlinear_B_matrix(
        self, H: np.ndarray, u_nodes: np.ndarray
    ) -> np.ndarray:
        """
        Computes the Total Lagrangian strain-displacement matrix B(u) = B_L + B_NL(u) (6 x 30).
        """
        BL = self.compute_linear_B_matrix(H)
        BNL = np.zeros((6, 30), dtype=np.float64)

        # Displacement gradient l = u^T @ H (shape: 3 x 3, l[i, j] = du_i / dX_j)
        l = u_nodes.T @ H
        du, dv, dw = l[0], l[1], l[2]

        for i in range(10):
            b, c, d = H[i, 0], H[i, 1], H[i, 2]
            col = i * 3

            # Normal strain derivatives
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
        self,
        node_coords: np.ndarray,
        u_nodes: np.ndarray,
        quadrature_order: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the Total Lagrangian tangent stiffness matrix k_tan (30x30)
        and internal force vector f_int (30,) via numerical quadrature.

        Parameters:
        -----------
        node_coords : np.ndarray of shape (10, 3)
            Reference coordinates X.
        u_nodes : np.ndarray of shape (10, 3)
            Current nodal displacements u.
        quadrature_order : int, optional
            Number of Gauss integration points (default: 4).

        Returns:
        --------
        k_tan : np.ndarray of shape (30, 30)
            Consistent tangent stiffness matrix: k_mat + k_geo.
        f_int : np.ndarray of shape (30,)
            Internal force vector.
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
            J = node_coords.T @ dN_dxi
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(
                    f"Negative or zero Jacobian determinant: det(J) = {detJ:.4e}."
                )

            H = dN_dxi @ np.linalg.inv(J)
            F = x.T @ H
            detF = np.linalg.det(F)
            if detF <= 0.0:
                raise ValueError(
                    f"Element inversion detected: det(F) = {detF:.4e} <= 0."
                )

            C, S = self._material.compute_stress_and_tangent(F)
            B = self.compute_nonlinear_B_matrix(H, u_nodes)
            G = self.compute_geometric_matrix(H)

            # Material tangent contribution
            k_mat = B.T @ C @ B

            # Geometric tangent contribution
            S_hat = np.kron(I3, S)
            k_geo = G.T @ S_hat @ G

            weight_detJ = qw[q] * detJ
            k_tan += (k_mat + k_geo) * weight_detJ

            # Internal force contribution
            S_voigt = tensor_to_voigt_stress(S)
            f_int += (B.T @ S_voigt) * weight_detJ

        return k_tan, f_int

